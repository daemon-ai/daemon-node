// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The `daemon-metta` worker binary.
//!
//! Speaks the length-framed [`daemon_metta::protocol`] over stdio (a [`CutChannel`]), exactly like
//! `daemon-infer`. Because the MeTTa runner is `!Send`, the binary is structured as:
//!
//! - an **actor thread** (`std::thread`) that constructs and owns the [`Worker`] (state + engine)
//!   and processes commands sequentially over a `std::sync::mpsc` channel, and
//! - the async stdio loop, which decodes inbound [`Command`] frames, forwards each to the actor with
//!   a `tokio::oneshot` reply, and writes the resulting [`Event`] frame back.
//!
//! The runner never crosses the thread boundary; only `Send` protocol frames + oneshot senders do.
//!
//! Usage: `daemon-metta [--state-dir <path>]` (no `--state-dir` => an ephemeral in-memory store).
//!
//! [`CutChannel`]: daemon_provision::CutChannel

#![forbid(unsafe_code)]

use daemon_metta::protocol::{self, Command, Event, Space};
use daemon_metta::state::MettaState;
use daemon_metta::worker::Worker;
use daemon_provision::CutChannel;
use tokio::sync::oneshot;

/// A request handed to the actor thread: the command plus the channel its reply is returned on.
type ActorMsg = (Command, oneshot::Sender<Option<Event>>);

#[tokio::main(flavor = "current_thread")]
async fn main() {
    // stdout is the cut transport, so all diagnostics go to stderr (eprintln!), like daemon-infer.
    let state_dir = parse_state_dir();
    let state = match &state_dir {
        Some(dir) => match MettaState::open(dir) {
            Ok(s) => s,
            Err(e) => {
                eprintln!(
                    "daemon-metta: failed to open state dir {}: {e}",
                    dir.display()
                );
                MettaState::in_memory()
            }
        },
        None => MettaState::in_memory(),
    };

    // Channels: the async loop -> actor (commands), and a one-shot startup handshake so the loop can
    // emit `Ready` with the actually-compiled engine + spaces.
    let (cmd_tx, cmd_rx) = std::sync::mpsc::channel::<ActorMsg>();
    let (ready_tx, ready_rx) = oneshot::channel::<String>();

    // The actor thread: owns the (`!Send`) Worker for its whole lifetime.
    let actor = std::thread::Builder::new()
        .name("metta-actor".into())
        .spawn(move || {
            let mut worker = Worker::new(state);
            let _ = ready_tx.send(worker.engine_name().to_string());
            while let Ok((cmd, reply)) = cmd_rx.recv() {
                let event = worker.handle(cmd);
                let stop = event.is_none();
                let _ = reply.send(event);
                if stop {
                    break;
                }
            }
        })
        .expect("spawn metta actor thread");

    let channel = CutChannel::from_stdio();
    let (writer, mut reader) = channel.split();

    // Startup: announce readiness with the compiled engine identifier.
    let engine = ready_rx.await.unwrap_or_else(|_| "unknown".into());
    let ready = Event::Ready {
        engine,
        spaces: Space::all()
            .iter()
            .map(|s| s.as_str().to_string())
            .collect(),
    };
    if let Ok(bytes) = protocol::encode(&ready) {
        let _ = writer.send(&bytes).await;
    }

    while let Some(bytes) = reader.recv().await {
        let cmd = match protocol::decode::<Command>(&bytes) {
            Ok(cmd) => cmd,
            Err(e) => {
                eprintln!("daemon-metta: undecodable command frame: {e}");
                continue;
            }
        };
        let is_shutdown = matches!(cmd, Command::Shutdown);

        let (reply_tx, reply_rx) = oneshot::channel();
        if cmd_tx.send((cmd, reply_tx)).is_err() {
            break; // actor gone
        }
        let event = match reply_rx.await {
            Ok(Some(event)) => event,
            Ok(None) => break, // shutdown acknowledged by the actor
            Err(_) => break,   // actor dropped the reply
        };
        match protocol::encode(&event) {
            Ok(frame) => {
                if writer.send(&frame).await.is_err() {
                    break;
                }
            }
            Err(e) => eprintln!("daemon-metta: failed to encode event: {e}"),
        }
        if is_shutdown {
            break;
        }
    }

    drop(cmd_tx);
    let _ = actor.join();
}

/// Parse `--state-dir <path>` from the argument vector.
fn parse_state_dir() -> Option<std::path::PathBuf> {
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--state-dir" {
            return args.next().map(std::path::PathBuf::from);
        }
        if let Some(rest) = arg.strip_prefix("--state-dir=") {
            return Some(std::path::PathBuf::from(rest));
        }
    }
    None
}
