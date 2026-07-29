//! Segment dump for live-run diagnosis: per-record one-liners over `scan_file`.
//!
//! Usage: `jdump <segment.dvhcjrn> [<segment…>]` — prints `ord tag-name` plus the discriminating
//! fields of the frame-bearing tags (event/publish/signed-frame/clock/timer), so a wedged
//! instance's last recorded protocol state is readable without the replay oracle.

use daemon_vhc_journal::{scan_file, Body};

fn hex8(b: &[u8]) -> String {
    b.iter().take(4).map(|x| format!("{x:02x}")).collect()
}

/// Best-effort peek at a frame's module message variant: walks the CBOR value tree and returns
/// the first map text key (an externally tagged enum's variant name), recursing through byte
/// strings that themselves parse as CBOR (the signed envelope nests payload bytes).
fn peek_kind(bytes: &[u8]) -> Option<String> {
    let v: ciborium::value::Value = ciborium::de::from_reader(bytes).ok()?;
    walk(&v, 0)
}

fn walk(v: &ciborium::value::Value, depth: usize) -> Option<String> {
    if depth > 6 {
        return None;
    }
    use ciborium::value::Value;
    match v {
        Value::Map(m) => {
            // An externally tagged enum is a single-entry map whose text key is the variant
            // name (capitalized). Struct field maps have lowercase keys — recurse through
            // their values instead.
            for (k, _) in m {
                if let Value::Text(t) = k {
                    if t.starts_with(char::is_uppercase) {
                        return Some(t.clone());
                    }
                }
            }
            m.iter().find_map(|(_, inner)| walk(inner, depth + 1))
        }
        Value::Array(items) => items.iter().find_map(|i| walk(i, depth + 1)),
        Value::Bytes(b) => ciborium::de::from_reader::<ciborium::value::Value, _>(b.as_slice())
            .ok()
            .and_then(|inner| walk(&inner, depth + 1)),
        _ => None,
    }
}

fn main() {
    for path in std::env::args().skip(1) {
        let scan = match scan_file(&path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("{path}: {e}");
                continue;
            }
        };
        println!("== {path} ({} records)", scan.records.len());
        for rec in &scan.records {
            let ord = rec.ord;
            match &rec.body {
                Body::RunHeader(_) => println!("{ord:>6} run-header"),
                Body::Event(e) => println!(
                    "{ord:>6} event at={} frame[{}]={}",
                    e.at,
                    e.frame.len(),
                    hex8(&e.frame)
                ),
                Body::ReadBack(_) => println!("{ord:>6} read-back"),
                Body::Clock(c) => println!("{ord:>6} clock now={}", c.now),
                Body::Publish(p) => println!(
                    "{ord:>6} publish ch={} seq={} frame[{}] kind={}",
                    p.channel,
                    p.seq,
                    p.frame.len(),
                    peek_kind(&p.frame).unwrap_or_else(|| "?".into())
                ),
                Body::TimerArm(t) => println!(
                    "{ord:>6} timer-arm id={} delay={} at={}",
                    t.id, t.delay, t.armed_at
                ),
                Body::TimerCancel(t) => println!("{ord:>6} timer-cancel id={}", t.id),
                Body::Drop(_) => println!("{ord:>6} drop"),
                Body::Throttle(_) => println!("{ord:>6} throttle"),
                Body::Terminal(t) => println!(
                    "{ord:>6} terminal kind={} outcome={:?} trap={:?}",
                    t.kind, t.outcome, t.trap
                ),
                Body::Condition(c) => println!("{ord:>6} condition {c:?}"),
                Body::Snapshot(_) => println!("{ord:>6} snapshot"),
                Body::Init(_) => println!("{ord:>6} init"),
                Body::SignedFrame(f) => println!(
                    "{ord:>6} signed-frame ch={} seq={} sender={} kind={}",
                    f.channel,
                    f.seq,
                    hex8(&f.sender.0),
                    f.frame
                        .as_deref()
                        .and_then(peek_kind)
                        .unwrap_or_else(|| "?".into())
                ),
                Body::Instantiation(_) => println!("{ord:>6} instantiation"),
                other => println!("{ord:>6} tag={}", other.tag()),
            }
        }
        if scan.truncated {
            println!("   (truncated tail)");
        }
    }
}
