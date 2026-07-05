// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Document-to-text extraction for the `fs` tool's `read` op (feature `extract`).
//!
//! A port of hermes `tools/read_extract.py` (`.ipynb` via JSON, `.docx` and `.xlsx` via zip +
//! streaming XML — hermes does all three with the python stdlib) plus the Cursor-parity PDF path
//! hermes does not have. Pure Rust throughout: `serde_json` (notebooks), `zip` + `quick-xml`
//! (DOCX + XLSX), `pdf-extract` (PDF). The XLSX parser is hand-rolled like hermes' rather than
//! using `calamine`: calamine's current release pins a `quick-xml` line carrying
//! RUSTSEC-2026-0194/0195, and the hermes-shape parser (shared strings + visible sheets + capped
//! rows) is all the read path needs. Extraction is attempted **before** the binary-extension
//! guard; a malformed document falls through to the normal "binary file" refusal, so failure
//! here is never fatal.

use std::collections::HashMap;
use std::io::Cursor;

use quick_xml::events::Event;

/// The input-size ceiling for any extraction (hermes caps XLSX at 50 MB; applied uniformly here
/// so a pathological document cannot pin the CPU).
const MAX_EXTRACT_BYTES: usize = 50 * 1024 * 1024;
/// Row cap per spreadsheet sheet (hermes `_MAX_XLSX_ROWS_PER_SHEET`).
const MAX_SHEET_ROWS: usize = 5000;
/// Column cap per spreadsheet row (hermes `_MAX_XLSX_COLS`).
const MAX_SHEET_COLS: usize = 256;

/// A supported-looking document could not be rendered as text (the caller falls back to the
/// binary-read refusal).
#[derive(Debug)]
pub struct ExtractError(pub String);

impl std::fmt::Display for ExtractError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// The document families the read path can extract.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DocKind {
    /// Jupyter notebook (`.ipynb`) — JSON cells rendered with cell headers.
    Notebook,
    /// Word document (`.docx`) — paragraph text from `word/document.xml`.
    Docx,
    /// Excel workbook (`.xlsx`/`.xlsm`) — visible sheets as tab-separated rows (hermes scope;
    /// legacy `.xls` / `.ods` are not extracted and fall back to the binary refusal).
    Spreadsheet,
    /// PDF (`.pdf`) — extracted text content.
    Pdf,
}

/// Classify `path` by extension. `None` = not an extractable document.
pub fn doc_kind(path: &str) -> Option<DocKind> {
    let ext = path.rsplit_once('.')?.1.to_ascii_lowercase();
    match ext.as_str() {
        "ipynb" => Some(DocKind::Notebook),
        "docx" => Some(DocKind::Docx),
        "xlsx" | "xlsm" => Some(DocKind::Spreadsheet),
        "pdf" => Some(DocKind::Pdf),
        _ => None,
    }
}

/// Extract `bytes` as readable text. Synchronous and potentially CPU-heavy (PDF/XLSX parsing) —
/// the caller runs it on a blocking thread.
pub fn extract_document_text(kind: DocKind, bytes: &[u8]) -> Result<String, ExtractError> {
    if bytes.len() > MAX_EXTRACT_BYTES {
        return Err(ExtractError(format!(
            "document too large to extract ({} bytes > {MAX_EXTRACT_BYTES})",
            bytes.len()
        )));
    }
    match kind {
        DocKind::Notebook => extract_notebook(bytes),
        DocKind::Docx => extract_docx(bytes),
        DocKind::Spreadsheet => extract_spreadsheet(bytes),
        DocKind::Pdf => extract_pdf(bytes),
    }
}

// ---------------------------------------------------------------------------------------------
// .ipynb (read_extract.py:61 `_extract_notebook`)
// ---------------------------------------------------------------------------------------------

/// A notebook cell's `source` is a string or a list of strings.
fn source_text(source: &serde_json::Value) -> String {
    match source {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Array(items) => items
            .iter()
            .filter_map(serde_json::Value::as_str)
            .collect::<String>(),
        _ => String::new(),
    }
}

fn extract_notebook(bytes: &[u8]) -> Result<String, ExtractError> {
    let nb: serde_json::Value = serde_json::from_slice(bytes)
        .map_err(|e| ExtractError(format!("not a valid notebook: {e}")))?;
    let obj = nb
        .as_object()
        .ok_or_else(|| ExtractError("notebook root is not an object".into()))?;

    // Modern shape: top-level `cells`; legacy shape: `worksheets[].cells`.
    let cells: Vec<&serde_json::Value> = match obj.get("cells").and_then(|c| c.as_array()) {
        Some(cells) => cells.iter().collect(),
        None => obj
            .get("worksheets")
            .and_then(|w| w.as_array())
            .map(|ws| {
                ws.iter()
                    .filter_map(|w| w.get("cells").and_then(|c| c.as_array()))
                    .flatten()
                    .collect()
            })
            .unwrap_or_default(),
    };
    if cells.is_empty() {
        return Err(ExtractError("notebook contains no cells".into()));
    }

    let mut counts = [0usize; 3]; // markdown, code, raw
    let mut out: Vec<String> = Vec::new();
    for cell in cells {
        let Some(cell) = cell.as_object() else {
            continue;
        };
        let (idx, label) = match cell.get("cell_type").and_then(|t| t.as_str()) {
            Some("markdown") => (0, "Markdown"),
            Some("code") => (1, "Code"),
            Some("raw") => (2, "Raw"),
            _ => continue,
        };
        counts[idx] += 1;
        let suffix = if label == "Raw" {
            String::new()
        } else {
            format!(" {}", counts[idx])
        };
        out.push(format!(
            "# \u{2500}\u{2500} {label} cell{suffix} \u{2500}\u{2500}"
        ));
        let body = cell.get("source").map(source_text).unwrap_or_default();
        out.push(body.trim_end_matches('\n').to_string());
        out.push(String::new());
    }
    if out.is_empty() {
        return Err(ExtractError("notebook contains no readable cells".into()));
    }
    Ok(format!("{}\n", out.join("\n").trim_end_matches('\n')))
}

// ---------------------------------------------------------------------------------------------
// .docx (read_extract.py:107 `_extract_docx`)
// ---------------------------------------------------------------------------------------------

/// The local part of a possibly-prefixed XML name (`w:t` → `t`).
fn local_name(qname: &[u8]) -> &[u8] {
    match qname.iter().rposition(|&b| b == b':') {
        Some(i) => &qname[i + 1..],
        None => qname,
    }
}

/// Read one named part out of a zip archive as (lossy) UTF-8 text.
fn zip_part(
    archive: &mut zip::ZipArchive<Cursor<&[u8]>>,
    name: &str,
) -> Result<String, ExtractError> {
    use std::io::Read as _;
    let mut part = archive
        .by_name(name)
        .map_err(|e| ExtractError(format!("missing {name}: {e}")))?;
    let mut raw = Vec::new();
    part.read_to_end(&mut raw)
        .map_err(|e| ExtractError(format!("reading {name}: {e}")))?;
    Ok(String::from_utf8_lossy(&raw).into_owned())
}

fn extract_docx(bytes: &[u8]) -> Result<String, ExtractError> {
    let mut archive = zip::ZipArchive::new(Cursor::new(bytes))
        .map_err(|e| ExtractError(format!("not a valid DOCX: {e}")))?;
    let document = zip_part(&mut archive, "word/document.xml")?;

    let mut reader = quick_xml::Reader::from_str(&document);
    let mut lines: Vec<String> = Vec::new();
    let mut para: Option<String> = None;
    let mut in_text = false;
    loop {
        match reader.read_event() {
            Ok(Event::Start(e)) => match local_name(e.name().as_ref()) {
                b"p" => para = Some(String::new()),
                b"t" => in_text = true,
                b"tab" => {
                    if let Some(p) = &mut para {
                        p.push('\t');
                    }
                }
                b"br" | b"cr" => {
                    if let Some(p) = &mut para {
                        p.push('\n');
                    }
                }
                _ => {}
            },
            Ok(Event::Empty(e)) => match local_name(e.name().as_ref()) {
                b"tab" => {
                    if let Some(p) = &mut para {
                        p.push('\t');
                    }
                }
                b"br" | b"cr" => {
                    if let Some(p) = &mut para {
                        p.push('\n');
                    }
                }
                _ => {}
            },
            Ok(Event::Text(t)) => {
                if in_text {
                    if let (Some(p), Ok(text)) = (&mut para, t.decode()) {
                        p.push_str(&text);
                    }
                }
            }
            Ok(Event::End(e)) => match local_name(e.name().as_ref()) {
                b"t" => in_text = false,
                b"p" => {
                    if let Some(p) = para.take() {
                        lines.extend(p.split('\n').map(str::to_string));
                    }
                }
                _ => {}
            },
            Ok(Event::Eof) => break,
            Ok(_) => {}
            Err(e) => return Err(ExtractError(format!("malformed XML in DOCX: {e}"))),
        }
    }
    if !lines.iter().any(|l| !l.trim().is_empty()) {
        return Err(ExtractError("DOCX contains no extractable text".into()));
    }
    Ok(format!("{}\n", lines.join("\n").trim_end_matches('\n')))
}

// ---------------------------------------------------------------------------------------------
// .xlsx (read_extract.py:133 `_extract_xlsx` — the same hand-rolled zip+XML walk hermes does)
// ---------------------------------------------------------------------------------------------

/// The attribute's (unescaped, lossy) text value, by local attribute name (`r:id` matches `id`).
fn attr_local(e: &quick_xml::events::BytesStart<'_>, local: &[u8]) -> Option<String> {
    for attr in e.attributes().flatten() {
        if local_name(attr.key.as_ref()) == local {
            return Some(String::from_utf8_lossy(&attr.value).into_owned());
        }
    }
    None
}

/// The shared-string table (`xl/sharedStrings.xml`): per `<si>`, every descendant `<t>` text
/// concatenated (hermes `_shared_strings`). Missing/malformed table = no shared strings.
fn shared_strings(archive: &mut zip::ZipArchive<Cursor<&[u8]>>) -> Vec<String> {
    let Ok(xml) = zip_part(archive, "xl/sharedStrings.xml") else {
        return Vec::new();
    };
    let mut reader = quick_xml::Reader::from_str(&xml);
    let mut out = Vec::new();
    let mut current: Option<String> = None;
    let mut in_t = false;
    loop {
        match reader.read_event() {
            Ok(Event::Start(e)) => match local_name(e.name().as_ref()) {
                b"si" => current = Some(String::new()),
                b"t" => in_t = current.is_some(),
                _ => {}
            },
            Ok(Event::Text(t)) => {
                if in_t {
                    if let (Some(buf), Ok(text)) = (&mut current, t.decode()) {
                        buf.push_str(&text);
                    }
                }
            }
            Ok(Event::End(e)) => match local_name(e.name().as_ref()) {
                b"t" => in_t = false,
                b"si" => {
                    if let Some(buf) = current.take() {
                        out.push(buf);
                    }
                }
                _ => {}
            },
            Ok(Event::Eof) => break,
            Ok(_) => {}
            Err(_) => return Vec::new(),
        }
    }
    out
}

/// The workbook's sheets in order: `(name, state, relationship id)` (hermes `_workbook_sheets`).
fn workbook_sheets(
    archive: &mut zip::ZipArchive<Cursor<&[u8]>>,
) -> Result<Vec<(String, String, String)>, ExtractError> {
    let xml = zip_part(archive, "xl/workbook.xml")?;
    let mut reader = quick_xml::Reader::from_str(&xml);
    let mut out = Vec::new();
    loop {
        match reader.read_event() {
            Ok(Event::Start(e) | Event::Empty(e)) if local_name(e.name().as_ref()) == b"sheet" => {
                out.push((
                    attr_local(&e, b"name").unwrap_or_else(|| "Sheet".into()),
                    attr_local(&e, b"state").unwrap_or_else(|| "visible".into()),
                    // `r:id` — match by local name; `sheetId` has a different local name.
                    attr_local(&e, b"id").unwrap_or_default(),
                ));
            }
            Ok(Event::Eof) => break,
            Ok(_) => {}
            Err(e) => return Err(ExtractError(format!("malformed xl/workbook.xml: {e}"))),
        }
    }
    Ok(out)
}

/// Workbook relationships: relationship id → target part (hermes `_workbook_rels`).
fn workbook_rels(archive: &mut zip::ZipArchive<Cursor<&[u8]>>) -> HashMap<String, String> {
    let Ok(xml) = zip_part(archive, "xl/_rels/workbook.xml.rels") else {
        return HashMap::new();
    };
    let mut reader = quick_xml::Reader::from_str(&xml);
    let mut out = HashMap::new();
    loop {
        match reader.read_event() {
            Ok(Event::Start(e) | Event::Empty(e))
                if local_name(e.name().as_ref()) == b"Relationship" =>
            {
                if let (Some(id), Some(target)) = (attr_local(&e, b"Id"), attr_local(&e, b"Target"))
                {
                    out.insert(id, target);
                }
            }
            Ok(Event::Eof) => break,
            Ok(_) => {}
            Err(_) => return HashMap::new(),
        }
    }
    out
}

/// Normalize a relationship target to a zip part path (hermes `_sheet_part`).
fn sheet_part(target: &str) -> String {
    let target = target.trim_start_matches('/');
    if target.starts_with("xl/") {
        target.to_string()
    } else {
        format!("xl/{target}")
    }
}

/// 0-based column index from a cell reference's letter prefix (hermes `_col_index`).
fn col_index(cell_ref: &str) -> usize {
    let mut idx: usize = 0;
    for c in cell_ref.chars() {
        if !c.is_ascii_alphabetic() {
            break;
        }
        idx = idx * 26 + (c.to_ascii_uppercase() as usize - 'A' as usize + 1);
    }
    idx.saturating_sub(1)
}

/// Resolve one cell's display value (hermes `_cell_value`): shared string, inline string,
/// boolean, error, or the raw `<v>` text.
fn cell_value(cell_type: &str, value: &str, inline: &str, shared: &[String]) -> String {
    match cell_type {
        "s" => value
            .parse::<usize>()
            .ok()
            .and_then(|i| shared.get(i).cloned())
            .unwrap_or_default(),
        "inlineStr" => inline.to_string(),
        "b" => {
            if matches!(value.trim(), "1" | "true" | "TRUE") {
                "TRUE".into()
            } else {
                "FALSE".into()
            }
        }
        "e" => {
            if value.is_empty() {
                "#ERROR".into()
            } else {
                value.to_string()
            }
        }
        _ => value.to_string(),
    }
}

/// Parse one sheet part into capped, tab-joinable rows (hermes `_sheet_rows`).
fn sheet_rows(xml: &str, shared: &[String]) -> Result<Vec<Vec<String>>, ExtractError> {
    let mut reader = quick_xml::Reader::from_str(xml);
    let mut rows: Vec<Vec<String>> = Vec::new();

    let mut cells: HashMap<usize, String> = HashMap::new();
    let mut max_col: Option<usize> = None;
    // In-cell state.
    let mut in_cell = false;
    let mut cell_col: usize = 0;
    let mut cell_type = String::new();
    let mut v_buf = String::new();
    let mut is_buf = String::new();
    let mut in_v = false;
    let mut in_is_t = false;

    macro_rules! close_cell {
        () => {
            if in_cell && cell_col < MAX_SHEET_COLS {
                cells.insert(cell_col, cell_value(&cell_type, &v_buf, &is_buf, shared));
                max_col = Some(max_col.map_or(cell_col, |m: usize| m.max(cell_col)));
            }
        };
    }

    macro_rules! open_cell {
        ($e:expr) => {{
            cell_col = attr_local($e, b"r")
                .map(|r| col_index(&r))
                .unwrap_or_else(|| max_col.map_or(0, |m| m + 1));
            cell_type = attr_local($e, b"t").unwrap_or_default();
            v_buf.clear();
            is_buf.clear();
            in_cell = true;
        }};
    }

    loop {
        if rows.len() >= MAX_SHEET_ROWS {
            break;
        }
        match reader.read_event() {
            Ok(Event::Start(e)) => match local_name(e.name().as_ref()) {
                b"row" => {
                    cells.clear();
                    max_col = None;
                }
                b"c" => open_cell!(&e),
                b"v" => in_v = in_cell,
                b"t" => in_is_t = in_cell && cell_type == "inlineStr",
                _ => {}
            },
            // A self-closing element gets no matching `End`: a `<c/>` is an empty cell that must
            // still occupy its column; a `<row/>` is an empty row.
            Ok(Event::Empty(e)) => match local_name(e.name().as_ref()) {
                b"c" => {
                    open_cell!(&e);
                    close_cell!();
                    in_cell = false;
                }
                b"row" => rows.push(Vec::new()),
                _ => {}
            },
            Ok(Event::Text(t)) => {
                if let Ok(text) = t.decode() {
                    if in_v {
                        v_buf.push_str(&text);
                    } else if in_is_t {
                        is_buf.push_str(&text);
                    }
                }
            }
            Ok(Event::End(e)) => match local_name(e.name().as_ref()) {
                b"v" => in_v = false,
                b"t" => in_is_t = false,
                b"c" => {
                    close_cell!();
                    in_cell = false;
                }
                b"row" => {
                    let row = match max_col {
                        Some(max) => (0..=max)
                            .map(|i| cells.get(&i).cloned().unwrap_or_default())
                            .collect(),
                        None => Vec::new(),
                    };
                    rows.push(row);
                }
                _ => {}
            },
            Ok(Event::Eof) => break,
            Ok(_) => {}
            Err(e) => return Err(ExtractError(format!("malformed sheet XML: {e}"))),
        }
    }
    // Trim trailing all-empty rows (hermes parity).
    while rows
        .last()
        .is_some_and(|r| r.iter().all(|v| v.trim().is_empty()))
    {
        rows.pop();
    }
    Ok(rows)
}

fn extract_spreadsheet(bytes: &[u8]) -> Result<String, ExtractError> {
    let mut archive = zip::ZipArchive::new(Cursor::new(bytes))
        .map_err(|e| ExtractError(format!("not a valid XLSX: {e}")))?;
    let shared = shared_strings(&mut archive);
    let sheets = workbook_sheets(&mut archive)?;
    let rels = workbook_rels(&mut archive);

    let mut out: Vec<String> = Vec::new();
    for (name, state, rid) in sheets {
        // Hidden / very-hidden sheets are skipped (hermes parity).
        if state == "hidden" || state == "veryHidden" {
            continue;
        }
        let Some(target) = rels.get(&rid) else {
            continue;
        };
        let Ok(xml) = zip_part(&mut archive, &sheet_part(target)) else {
            continue;
        };
        let Ok(rows) = sheet_rows(&xml, &shared) else {
            continue;
        };
        out.push(format!("# \u{2500}\u{2500} Sheet: {name} \u{2500}\u{2500}"));
        if rows.is_empty() {
            out.push("(empty)".to_string());
        } else {
            out.extend(rows.iter().map(|row| row.join("\t")));
        }
        out.push(String::new());
    }
    if out.is_empty() {
        return Err(ExtractError(
            "XLSX has no visible sheets with content".into(),
        ));
    }
    Ok(format!("{}\n", out.join("\n").trim_end_matches('\n')))
}

// ---------------------------------------------------------------------------------------------
// .pdf (Cursor-parity addition; hermes has no PDF extraction)
// ---------------------------------------------------------------------------------------------

fn extract_pdf(bytes: &[u8]) -> Result<String, ExtractError> {
    let text = pdf_extract::extract_text_from_mem(bytes)
        .map_err(|e| ExtractError(format!("PDF extraction failed: {e}")))?;
    if !text.chars().any(|c| !c.is_whitespace()) {
        return Err(ExtractError("PDF contains no extractable text".into()));
    }
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kind_classification() {
        assert_eq!(doc_kind("nb.ipynb"), Some(DocKind::Notebook));
        assert_eq!(doc_kind("a/b.DOCX"), Some(DocKind::Docx));
        assert_eq!(doc_kind("x.xlsx"), Some(DocKind::Spreadsheet));
        assert_eq!(doc_kind("x.xlsm"), Some(DocKind::Spreadsheet));
        assert_eq!(doc_kind("p.pdf"), Some(DocKind::Pdf));
        // Legacy .xls / .ods are hermes-scope-out: they fall back to the binary refusal.
        assert_eq!(doc_kind("x.ods"), None);
        assert_eq!(doc_kind("x.xls"), None);
        assert_eq!(doc_kind("code.rs"), None);
        assert_eq!(doc_kind("noext"), None);
    }

    #[test]
    fn notebook_renders_cells_with_headers() {
        let nb = serde_json::json!({
            "cells": [
                {"cell_type": "markdown", "source": ["# Title\n", "text\n"]},
                {"cell_type": "code", "source": "print('hi')\n"},
                {"cell_type": "raw", "source": "raw stuff"},
            ]
        });
        let out = extract_notebook(serde_json::to_vec(&nb).unwrap().as_slice()).unwrap();
        assert!(out.contains("# \u{2500}\u{2500} Markdown cell 1 \u{2500}\u{2500}\n# Title\ntext"));
        assert!(out.contains("# \u{2500}\u{2500} Code cell 1 \u{2500}\u{2500}\nprint('hi')"));
        assert!(out.contains("# \u{2500}\u{2500} Raw cell \u{2500}\u{2500}\nraw stuff"));
    }

    #[test]
    fn notebook_rejects_malformed() {
        assert!(extract_notebook(b"not json").is_err());
        assert!(extract_notebook(b"[]").is_err());
        assert!(extract_notebook(br#"{"cells": []}"#).is_err());
    }

    /// Build a minimal in-memory zip from `(part name, content)` pairs.
    fn tiny_zip(parts: &[(&str, &str)]) -> Vec<u8> {
        use std::io::Write as _;
        let mut buf = Cursor::new(Vec::new());
        {
            let mut writer = zip::ZipWriter::new(&mut buf);
            for (name, content) in parts {
                writer
                    .start_file(*name, zip::write::SimpleFileOptions::default())
                    .unwrap();
                writer.write_all(content.as_bytes()).unwrap();
            }
            writer.finish().unwrap();
        }
        buf.into_inner()
    }

    /// Build a minimal in-memory DOCX (one zip entry) for the parser test.
    fn tiny_docx(document_xml: &str) -> Vec<u8> {
        tiny_zip(&[("word/document.xml", document_xml)])
    }

    #[test]
    fn docx_extracts_paragraphs_tabs_and_breaks() {
        let xml = r#"<?xml version="1.0"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>Hello</w:t><w:tab/><w:t>world</w:t></w:r></w:p>
    <w:p><w:r><w:t>line one</w:t><w:br/><w:t>line two</w:t></w:r></w:p>
  </w:body>
</w:document>"#;
        let out = extract_docx(&tiny_docx(xml)).unwrap();
        assert_eq!(out, "Hello\tworld\nline one\nline two\n");
    }

    #[test]
    fn docx_rejects_garbage_and_empty() {
        assert!(extract_docx(b"not a zip").is_err());
        let xml = r#"<w:document xmlns:w="http://x"><w:body><w:p/></w:body></w:document>"#;
        assert!(extract_docx(&tiny_docx(xml)).is_err(), "no text -> error");
    }

    /// A two-sheet workbook: one visible (shared string + inline string + number + boolean +
    /// skipped-column cell), one hidden (must not render).
    fn tiny_xlsx() -> Vec<u8> {
        let workbook = r#"<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
    <sheet name="Data" sheetId="1" r:id="rId1"/>
    <sheet name="Secret" sheetId="2" state="hidden" r:id="rId2"/>
  </sheets>
</workbook>"#;
        let rels = r#"<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="t" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="t" Target="worksheets/sheet2.xml"/>
</Relationships>"#;
        let shared = r#"<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><si><t>Name</t></si><si><t>Ada</t></si></sst>"#;
        let sheet1 = r#"<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetData>
    <row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="inlineStr"><is><t>Age</t></is></c></row>
    <row r="2"><c r="A2" t="s"><v>1</v></c><c r="C2"><v>36</v></c><c r="D2" t="b"><v>1</v></c></row>
  </sheetData>
</worksheet>"#;
        let sheet2 = r#"<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row r="1"><c r="A1"><v>hidden!</v></c></row></sheetData></worksheet>"#;
        tiny_zip(&[
            ("xl/workbook.xml", workbook),
            ("xl/_rels/workbook.xml.rels", rels),
            ("xl/sharedStrings.xml", shared),
            ("xl/worksheets/sheet1.xml", sheet1),
            ("xl/worksheets/sheet2.xml", sheet2),
        ])
    }

    #[test]
    fn xlsx_extracts_visible_sheets_with_cell_types() {
        let out = extract_spreadsheet(&tiny_xlsx()).unwrap();
        assert!(
            out.contains("# \u{2500}\u{2500} Sheet: Data \u{2500}\u{2500}"),
            "{out}"
        );
        assert!(out.contains("Name\tAge"), "shared + inline strings: {out}");
        // Row 2: A=shared "Ada", B skipped (empty), C=36, D=TRUE.
        assert!(out.contains("Ada\t\t36\tTRUE"), "{out}");
        assert!(!out.contains("Secret"), "hidden sheet leaked: {out}");
        assert!(!out.contains("hidden!"), "hidden sheet data leaked: {out}");
    }

    #[test]
    fn xlsx_rejects_garbage_and_empty_workbooks() {
        assert!(extract_spreadsheet(b"not a zip").is_err());
        let no_sheets = tiny_zip(&[(
            "xl/workbook.xml",
            r#"<workbook xmlns="http://x"><sheets/></workbook>"#,
        )]);
        assert!(extract_spreadsheet(&no_sheets).is_err());
    }

    #[test]
    fn xlsx_column_index_math() {
        assert_eq!(col_index("A1"), 0);
        assert_eq!(col_index("D2"), 3);
        assert_eq!(col_index("Z9"), 25);
        assert_eq!(col_index("AA10"), 26);
    }

    #[test]
    fn pdf_rejects_garbage() {
        assert!(extract_pdf(b"%PDF-9.9 garbage").is_err());
    }

    #[test]
    fn oversized_input_is_refused_cheaply() {
        // The size gate fires before any parsing; use a fake length via capacity trick is not
        // possible, so this just documents the constant's presence with a small direct call.
        let big = vec![0u8; MAX_EXTRACT_BYTES + 1];
        assert!(extract_document_text(DocKind::Pdf, &big).is_err());
    }
}
