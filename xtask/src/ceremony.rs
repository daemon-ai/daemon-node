// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `xtask author-ceremony-genesis` — the fleet-ceremony genesis authoring executable.
//!
//! A THIN wrapper around the frozen, reviewed, tested library
//! ([`daemon_vhc_testkit::ceremony::ceremony_genesis`]) — it never reimplements the authoring.
//! Its job is only to marshal the operator's ceremony-time inputs (the published corpus manifest,
//! the fleet's certified base identities, the built module hashes, the calibrated real run timers)
//! into a [`CeremonyGenesisSpec`], freeze the genesis with the author key, and write the four
//! operator artifacts:
//!
//! - `envelope.cbor` — the canonical frozen envelope bytes (the object the registry stores; the
//!   run id is their blake3);
//! - `envelope.b64` — base64 of those exact bytes, for the cloud seeder's `VHC_ENVELOPE_B64`;
//! - `run-id.txt` — the genesis hash hex (the cryptographic run id);
//! - `authoring-report.txt` — every frozen pin restated for human ratification.
//!
//! It also authors single-peer smoke-run geneses (min=max=1, small `--stop-rounds`): nothing here
//! hardcodes fleet-only assumptions beyond what the library itself enforces (geometry, chunk
//! divisibility, cadence↔retention).

use std::path::PathBuf;

use anyhow::{Context, Result};
use base64::Engine as _;

use daemon_vhc_proto::corpus::CorpusManifest;
use daemon_vhc_proto::{blake3_hash, Hash, PeerId, SigningKey};
use daemon_vhc_testkit::ceremony::{
    ceremony_expected_state_root, ceremony_genesis, ceremony_profile_chunk,
    ceremony_state_chunk_size, CeremonyGenesisSpec, CeremonyRunTimers, CEREMONY_EXPECTED_ROOT,
    CEREMONY_PARAM_COUNT, CEREMONY_SEQ_LEN,
};

/// The parsed `author-ceremony-genesis` inputs (see the [`crate::Cmd`] arm docs for each flag).
pub struct Args {
    pub run_label: String,
    pub author_key: String,
    pub coordinator_module: String,
    pub trainer_module: String,
    pub corpus_manifest: PathBuf,
    pub trusted_base: Vec<String>,
    pub roster: Vec<String>,
    pub upgrade_authority: Vec<String>,
    pub min_peers: u32,
    pub max_peers: u32,
    pub ckpt_cadence: u64,
    pub payload_retention: u64,
    pub warmup_s: u64,
    pub round_max_s: u64,
    pub witness_s: u64,
    pub cooldown_s: u64,
    pub stop_rounds: u64,
    pub out: PathBuf,
}

/// Parse 32 lowercase/uppercase hex bytes into a fixed array.
fn parse_hex32(what: &str, s: &str) -> Result<[u8; 32]> {
    let s = s.trim();
    anyhow::ensure!(
        s.len() == 64,
        "{what}: expected 64 hex chars (32 bytes), got {}",
        s.len()
    );
    let mut out = [0u8; 32];
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = u8::from_str_radix(&s[i * 2..i * 2 + 2], 16)
            .with_context(|| format!("{what}: invalid hex"))?;
    }
    Ok(out)
}

fn parse_hash(what: &str, s: &str) -> Result<Hash> {
    Ok(Hash::new(parse_hex32(what, s)?))
}

fn parse_peer(what: &str, s: &str) -> Result<PeerId> {
    Ok(PeerId::new(parse_hex32(what, s)?))
}

/// Load the author signing key from a `file` (32 raw bytes, or 64-hex text) or a bare 64-hex arg.
fn load_signing_key(arg: &str) -> Result<SigningKey> {
    let path = PathBuf::from(arg);
    let bytes: [u8; 32] = if path.is_file() {
        let raw = std::fs::read(&path).with_context(|| format!("read author key {arg}"))?;
        if raw.len() == 32 {
            raw.try_into().expect("checked len 32")
        } else {
            let text = String::from_utf8(raw).with_context(|| {
                format!("author key {arg} is neither 32 raw bytes nor hex text")
            })?;
            parse_hex32("author key file", text.trim())?
        }
    } else {
        parse_hex32("author key", arg)?
    };
    Ok(SigningKey::from_bytes(&bytes))
}

/// Author, freeze, and write the ceremony genesis artifacts.
pub fn run(args: Args) -> Result<()> {
    // -- resolve the ceremony-time inputs ---------------------------------------------------------
    let author = load_signing_key(&args.author_key)?;
    let coordinator_module = parse_hash("--coordinator-module", &args.coordinator_module)?;
    let trainer_module = parse_hash("--trainer-module", &args.trainer_module)?;

    let trusted_bases: Vec<PeerId> = args
        .trusted_base
        .iter()
        .map(|s| parse_peer("--trusted-base", s))
        .collect::<Result<_>>()?;
    let roster: Vec<PeerId> = args
        .roster
        .iter()
        .map(|s| parse_peer("--roster", s))
        .collect::<Result<_>>()?;
    let upgrade_authority: Vec<PeerId> = args
        .upgrade_authority
        .iter()
        .map(|s| parse_peer("--upgrade-authority", s))
        .collect::<Result<_>>()?;

    anyhow::ensure!(
        !trusted_bases.is_empty(),
        "at least one --trusted-base is required (the first is the coordinator authority)"
    );

    // -- the corpus pin + its artifact list, derived from the published manifest ------------------
    let manifest_bytes = std::fs::read(&args.corpus_manifest)
        .with_context(|| format!("read corpus manifest {}", args.corpus_manifest.display()))?;
    let manifest = CorpusManifest::from_canonical_bytes(&manifest_bytes)
        .map_err(|e| anyhow::anyhow!("parse corpus manifest: {e}"))?;
    let corpus_manifest_hash = blake3_hash(&manifest_bytes);

    anyhow::ensure!(
        u64::from(manifest.seq_len) == u64::from(CEREMONY_SEQ_LEN),
        "corpus seq_len {} != the frozen ceremony seq_len {CEREMONY_SEQ_LEN} — the corpus must be \
         tokenized at the frozen sequence length",
        manifest.seq_len
    );

    // The trainer role's `data@2` fetch grants: the manifest + the tokenizer + every shard, each by
    // its content/fold identity — exactly the set `publish-corpus` uploads and prints.
    let mut corpus_artifacts: Vec<(String, Hash)> = vec![
        ("corpus-manifest.cbor".to_string(), corpus_manifest_hash),
        ("tokenizer.json".to_string(), manifest.tokenizer.hash),
    ];
    for (i, shard) in manifest.shards.iter().enumerate() {
        corpus_artifacts.push((format!("shard-{i}.bin"), shard.shard_hash));
    }

    let timers = CeremonyRunTimers {
        warmup_s: args.warmup_s,
        round_max_s: args.round_max_s,
        witness_s: args.witness_s,
        cooldown_s: args.cooldown_s,
        stop_rounds: args.stop_rounds,
    };

    let spec = CeremonyGenesisSpec {
        run_label: &args.run_label,
        coordinator_module,
        trainer_module,
        corpus_manifest: corpus_manifest_hash,
        corpus_artifacts: &corpus_artifacts,
        seq_len: u64::from(manifest.seq_len),
        trusted_bases: &trusted_bases,
        roster: &roster,
        upgrade_authority: upgrade_authority.clone(),
        min_peers: args.min_peers,
        max_peers: args.max_peers,
        remote_ckpt_cadence_rounds: args.ckpt_cadence,
        payload_retention_rounds: args.payload_retention,
        timers,
    };

    // -- freeze (the library validates geometry, chunk divisibility, cadence↔retention) -----------
    let frozen = ceremony_genesis(&spec, &author)
        .map_err(|e| anyhow::anyhow!("author ceremony genesis: {e}"))?;

    // Belt-and-suspenders: re-open the frozen bytes (re-derives the hash + verifies the signature)
    // and validate the envelope, so the artifacts we write are provably the ones that re-open.
    let reopened = daemon_vhc_proto::FrozenGenesis::open(
        frozen.bytes().to_vec(),
        *frozen.signature(),
        *frozen.signer(),
    )
    .map_err(|e| anyhow::anyhow!("re-open the frozen genesis: {e}"))?;
    anyhow::ensure!(
        reopened.run_id() == frozen.run_id(),
        "re-opened run id does not match the frozen run id"
    );

    // The seed-derived matched-init root the peers cross-check — reproduced here so the report
    // states a computed value, not a transcribed constant.
    let reproduced_root = ceremony_expected_state_root();
    anyhow::ensure!(
        reproduced_root == Hash::new(CEREMONY_EXPECTED_ROOT),
        "reproduced expected_root does not match the pinned constant — refusing to author"
    );

    // -- write the four operator artifacts --------------------------------------------------------
    std::fs::create_dir_all(&args.out)
        .with_context(|| format!("create --out dir {}", args.out.display()))?;
    let run_id_hex = frozen.run_id().to_hex();
    let b64 = base64::engine::general_purpose::STANDARD.encode(frozen.bytes());

    std::fs::write(args.out.join("envelope.cbor"), frozen.bytes())
        .context("write envelope.cbor")?;
    std::fs::write(args.out.join("envelope.b64"), &b64).context("write envelope.b64")?;
    std::fs::write(args.out.join("run-id.txt"), format!("{run_id_hex}\n"))
        .context("write run-id.txt")?;

    let report = authoring_report(&AuthoringReport {
        run_label: &args.run_label,
        run_id_hex: &run_id_hex,
        signer_hex: frozen.signer().to_hex(),
        signature_hex: frozen.signature().to_hex(),
        coordinator_module,
        trainer_module,
        corpus_manifest_hash,
        manifest: &manifest,
        trusted_bases: &trusted_bases,
        roster: &roster,
        upgrade_authority: &upgrade_authority,
        min_peers: args.min_peers,
        max_peers: args.max_peers,
        ckpt_cadence: args.ckpt_cadence,
        payload_retention: args.payload_retention,
        timers,
        reproduced_root,
    });
    std::fs::write(args.out.join("authoring-report.txt"), &report)
        .context("write authoring-report.txt")?;

    println!("authored ceremony genesis:");
    println!("  run id  = {run_id_hex}");
    println!("  out dir = {}", args.out.display());
    println!("  wrote envelope.cbor, envelope.b64, run-id.txt, authoring-report.txt");
    print!("\n{report}");
    Ok(())
}

struct AuthoringReport<'a> {
    run_label: &'a str,
    run_id_hex: &'a str,
    signer_hex: String,
    signature_hex: String,
    coordinator_module: Hash,
    trainer_module: Hash,
    corpus_manifest_hash: Hash,
    manifest: &'a CorpusManifest,
    trusted_bases: &'a [PeerId],
    roster: &'a [PeerId],
    upgrade_authority: &'a [PeerId],
    min_peers: u32,
    max_peers: u32,
    ckpt_cadence: u64,
    payload_retention: u64,
    timers: CeremonyRunTimers,
    reproduced_root: Hash,
}

/// Restate every frozen pin for human ratification. Semantic names only — no plan codenames.
fn authoring_report(r: &AuthoringReport<'_>) -> String {
    use std::fmt::Write as _;
    let mut s = String::new();
    let _ = writeln!(s, "VHC fleet-ceremony genesis — authoring report");
    let _ = writeln!(s, "=============================================");
    let _ = writeln!(s, "run label            : {}", r.run_label);
    let _ = writeln!(s, "run id (genesis hash): {}", r.run_id_hex);
    let _ = writeln!(s, "author signer        : {}", r.signer_hex);
    let _ = writeln!(s, "author signature     : {}", r.signature_hex);
    let _ = writeln!(s);
    let _ = writeln!(
        s,
        "-- frozen model geometry (from the reviewed ceremony module) --"
    );
    let _ = writeln!(s, "param count          : {CEREMONY_PARAM_COUNT}");
    let _ = writeln!(
        s,
        "expected state root  : {}  (reproduced from the seed init at authoring)",
        r.reproduced_root.to_hex()
    );
    let _ = writeln!(s, "profile chunk        : {}", ceremony_profile_chunk());
    let _ = writeln!(
        s,
        "state chunk size     : {} bytes",
        ceremony_state_chunk_size()
    );
    let _ = writeln!(s, "sequence length      : {}", CEREMONY_SEQ_LEN);
    let _ = writeln!(s);
    let _ = writeln!(s, "-- modules --");
    let _ = writeln!(
        s,
        "coordinator.wasm     : {}",
        r.coordinator_module.to_hex()
    );
    let _ = writeln!(s, "worker.wasm          : {}", r.trainer_module.to_hex());
    let _ = writeln!(s);
    let _ = writeln!(s, "-- corpus --");
    let _ = writeln!(
        s,
        "manifest hash (pin)  : {}",
        r.corpus_manifest_hash.to_hex()
    );
    let _ = writeln!(
        s,
        "tokenizer hash       : {}",
        r.manifest.tokenizer.hash.to_hex()
    );
    let _ = writeln!(s, "tokenizer name       : {}", r.manifest.tokenizer.name);
    let _ = writeln!(
        s,
        "seq_len / chunk_size : {} / {}",
        r.manifest.seq_len, r.manifest.chunk_size
    );
    let _ = writeln!(s, "total tokens         : {}", r.manifest.total_tokens);
    let _ = writeln!(s, "shards               : {}", r.manifest.shards.len());
    for (i, shard) in r.manifest.shards.iter().enumerate() {
        let _ = writeln!(
            s,
            "  shard {i:<3}          : {}  ({} tokens)",
            shard.shard_hash.to_hex(),
            shard.token_count
        );
    }
    let _ = writeln!(s);
    let _ = writeln!(
        s,
        "-- trust set (ordered; first = coordinator authority) --"
    );
    for (i, p) in r.trusted_bases.iter().enumerate() {
        let _ = writeln!(s, "  trusted base {i}     : {}", p.to_hex());
    }
    let _ = writeln!(s, "-- roster (trainer assignment) --");
    for (i, p) in r.roster.iter().enumerate() {
        let _ = writeln!(s, "  roster {i}           : {}", p.to_hex());
    }
    let _ = writeln!(s, "-- upgrade authority --");
    if r.upgrade_authority.is_empty() {
        let _ = writeln!(
            s,
            "  (none — the run is immutable; no module upgrade is authorized)"
        );
    } else {
        for (i, p) in r.upgrade_authority.iter().enumerate() {
            let _ = writeln!(s, "  authority {i}        : {}", p.to_hex());
        }
    }
    let _ = writeln!(s);
    let _ = writeln!(s, "-- membership + retention --");
    let _ = writeln!(
        s,
        "min / max peers      : {} / {}",
        r.min_peers, r.max_peers
    );
    let _ = writeln!(s, "checkpoint cadence   : {} rounds", r.ckpt_cadence);
    let _ = writeln!(s, "payload retention    : {} rounds", r.payload_retention);
    let _ = writeln!(s);
    let _ = writeln!(s, "-- real run timers (calibrated) --");
    let _ = writeln!(s, "warmup               : {} s", r.timers.warmup_s);
    let _ = writeln!(s, "round train max      : {} s", r.timers.round_max_s);
    let _ = writeln!(s, "round witness        : {} s", r.timers.witness_s);
    let _ = writeln!(s, "cooldown             : {} s", r.timers.cooldown_s);
    let _ = writeln!(s, "stop after           : {} rounds", r.timers.stop_rounds);
    let _ = writeln!(s);
    let _ = writeln!(
        s,
        "cadence check        : PASSED at authoring (cadence + one churn slot <= retention)"
    );
    s
}
