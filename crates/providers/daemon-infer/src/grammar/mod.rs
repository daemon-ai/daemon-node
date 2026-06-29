// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! MeTTa grammar artifacts for grammar-constrained generation.
//!
//! The supplied MeTTa EBNF, transcribed into the two formats the local engines accept:
//! - [`METTA_GBNF`] — GBNF for llama.cpp (`llama-cpp-4` `LlamaSampler::grammar`).
//! - [`METTA_LARK`] — Lark for mistral.rs (`Constraint::Lark`, llguidance dialect).
//!
//! Both encode the same language. The engine-facing artifacts can only be exercised end-to-end with
//! the engines compiled in (the feature-gated worker lanes), so the unit tests here validate the
//! *language* against a small pure-Rust reference recognizer ([`recognize`]) — the oracle the two
//! grammars are checked against — plus basic non-emptiness/shape checks on the artifacts themselves.

/// The GBNF grammar (llama.cpp). Its root rule is `root`.
pub const METTA_GBNF: &str = include_str!("metta.gbnf");

/// The Lark grammar (mistral.rs / llguidance). Its start rule is `start`.
pub const METTA_LARK: &str = include_str!("metta.lark");

/// The GBNF root rule name.
pub const GBNF_ROOT: &str = "root";

/// Whether `c` is MeTTa whitespace.
fn is_ws(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\r' | '\n')
}

/// A pure-Rust reference recognizer for the MeTTa EBNF — the oracle the GBNF/Lark are validated
/// against. Returns `true` iff `src` is a well-formed MeTTa program (a sequence of optionally
/// `!`-prefixed atoms separated by whitespace/comments).
pub fn recognize(src: &str) -> bool {
    let chars: Vec<char> = src.chars().collect();
    let mut p = Recognizer { chars, pos: 0 };
    p.skip_delim();
    while p.pos < p.chars.len() {
        if p.peek() == Some('!') {
            p.pos += 1;
            p.skip_delim();
        }
        if !p.atom() {
            return false;
        }
        p.skip_delim();
    }
    true
}

struct Recognizer {
    chars: Vec<char>,
    pos: usize,
}

impl Recognizer {
    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    /// DELIM: a possibly-empty run of whitespace and `;` line comments.
    fn skip_delim(&mut self) {
        loop {
            match self.peek() {
                Some(c) if is_ws(c) => self.pos += 1,
                Some(';') => {
                    while let Some(c) = self.peek() {
                        self.pos += 1;
                        if c == '\n' {
                            break;
                        }
                    }
                }
                _ => break,
            }
        }
    }

    fn atom(&mut self) -> bool {
        match self.peek() {
            Some('(') => self.expression(),
            Some('$') => self.variable(),
            Some('"') => self.string(),
            Some(c) if !is_ws(c) && !matches!(c, ')' | ';') => self.word(),
            _ => false,
        }
    }

    /// EXPRESSION ::= '(' { ATOM [ DELIM ] } ')'
    fn expression(&mut self) -> bool {
        if self.peek() != Some('(') {
            return false;
        }
        self.pos += 1;
        self.skip_delim();
        loop {
            match self.peek() {
                Some(')') => {
                    self.pos += 1;
                    return true;
                }
                None => return false, // unterminated
                _ => {
                    if !self.atom() {
                        return false;
                    }
                    self.skip_delim();
                }
            }
        }
    }

    /// WORD ::= ( CHAR | '#' ) { CHAR | '"' | '#' }
    fn word(&mut self) -> bool {
        // First char: CHAR or '#'  ->  not ws, not one of '"' '(' ')' ';'.
        match self.peek() {
            Some(c) if !is_ws(c) && !matches!(c, '"' | '(' | ')' | ';') => self.pos += 1,
            _ => return false,
        }
        // Rest: CHAR or '"' or '#'  ->  not ws, not one of '(' ')' ';'.
        while let Some(c) = self.peek() {
            if !is_ws(c) && !matches!(c, '(' | ')' | ';') {
                self.pos += 1;
            } else {
                break;
            }
        }
        true
    }

    /// VARIABLE ::= '$' ( CHAR | '"' ) { CHAR | '"' }
    fn variable(&mut self) -> bool {
        if self.peek() != Some('$') {
            return false;
        }
        self.pos += 1;
        // At least one var char: CHAR or '"'  ->  not ws, not one of '#' '(' ')' ';'.
        let mut count = 0;
        while let Some(c) = self.peek() {
            if !is_ws(c) && !matches!(c, '#' | '(' | ')' | ';') {
                self.pos += 1;
                count += 1;
            } else {
                break;
            }
        }
        count > 0
    }

    /// STRING ::= '"' { any char except '"' } '"'
    fn string(&mut self) -> bool {
        if self.peek() != Some('"') {
            return false;
        }
        self.pos += 1;
        while let Some(c) = self.peek() {
            self.pos += 1;
            if c == '"' {
                return true;
            }
        }
        false // unterminated
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const VALID: &[&str] = &[
        "(owns alice artifact-42)",
        "(= (double $x) (* 2 $x))",
        "! (eval (+ 1 2))",
        "(prefers user \"direct style\")",
        "; a leading comment\n(fact a)",
        "(nested (a (b (c $x))))",
        "$x",
        "symbol",
        "(a) (b) (c)",
        "(tag #hashy)",
    ];

    const INVALID: &[&str] = &[
        "(unbalanced",
        "balanced)",
        "(a))",
        "\"unterminated string",
        "$",      // variable needs at least one char
        "(a (b)", // missing close
    ];

    #[test]
    fn recognizer_accepts_valid_metta() {
        for s in VALID {
            assert!(recognize(s), "should accept: {s:?}");
        }
    }

    #[test]
    fn recognizer_rejects_invalid_metta() {
        for s in INVALID {
            assert!(!recognize(s), "should reject: {s:?}");
        }
    }

    #[test]
    fn artifacts_are_present_and_shaped() {
        assert!(
            METTA_GBNF.contains("root"),
            "gbnf must define the root rule"
        );
        assert!(METTA_GBNF.contains("expression"));
        assert!(
            METTA_LARK.contains("start:"),
            "lark must define the start rule"
        );
        assert!(METTA_LARK.contains("VARIABLE"));
        // The artifacts are non-trivial.
        assert!(METTA_GBNF.len() > 200);
        assert!(METTA_LARK.len() > 200);
    }
}
