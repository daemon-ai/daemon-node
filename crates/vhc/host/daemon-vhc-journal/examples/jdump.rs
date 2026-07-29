//! Segment dump for live-run diagnosis: per-record one-liners over `scan_file`.
//!
//! Usage: `jdump <segment.dvhcjrn> [<segment…>]` — prints `ord tag-name` plus the discriminating
//! fields of the frame-bearing tags (event/publish/signed-frame/clock/timer), so a wedged
//! instance's last recorded protocol state is readable without the replay oracle.

use daemon_vhc_journal::{scan_file, Body};

fn hex8(b: &[u8]) -> String {
    b.iter().take(4).map(|x| format!("{x:02x}")).collect()
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
                    "{ord:>6} publish ch={} seq={} frame[{}]",
                    p.channel,
                    p.seq,
                    p.frame.len()
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
                    "{ord:>6} signed-frame ch={} seq={} sender={}",
                    f.channel,
                    f.seq,
                    hex8(&f.sender.0)
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
