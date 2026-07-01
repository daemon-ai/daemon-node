// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The `daemon-cli` command grammar: the clap `Parser`/`Subcommand` types. These are pure argument
//! definitions (no transport or `daemon_api` types) — the handlers in [`crate::cmd`] map a parsed
//! command into one [`daemon_api::ApiRequest`].

use std::path::PathBuf;

use clap::{Parser, Subcommand};

/// Operate a running `daemon` host node over its api socket.
#[derive(Parser)]
#[command(name = "daemon-cli", version = daemon_common::VERSION, about)]
pub(crate) struct Cli {
    /// Path to the node's api socket (defaults to `$DAEMON_SOCKET_PATH` or `$TMPDIR/daemon-api.sock`).
    #[arg(long, global = true)]
    pub(crate) socket: Option<PathBuf>,
    #[command(subcommand)]
    pub(crate) command: Command,
}

#[derive(Subcommand)]
pub(crate) enum Command {
    /// Print resident-service health and durable stats together.
    Status,
    /// Resident-service tree health.
    Health,
    /// Durable queue depths and session/active counts.
    Stats,
    /// Telemetry dump: folded usage/cost + event count + health + queue depths.
    Telemetry,
    /// List durable sessions and their states.
    Sessions,
    /// Create-if-absent and wake a durable session.
    Assign {
        /// The session id.
        id: String,
    },
    /// Cancel in-flight work for a session.
    Cancel {
        /// The session id.
        id: String,
    },
    /// Rename a session (set/clear its roster title).
    Rename {
        /// The session id.
        id: String,
        /// The new title (omit to clear it).
        title: Option<String>,
    },
    /// Pin or unpin a session (pinned conversations sort first in the roster).
    Pin {
        /// The session id.
        id: String,
        /// Unpin instead of pin.
        #[arg(long)]
        off: bool,
    },
    /// Archive or unarchive a session (archived conversations leave the default roster).
    Archive {
        /// The session id.
        id: String,
        /// Unarchive instead of archive.
        #[arg(long)]
        off: bool,
    },
    /// The orchestration fleet roster + folded usage.
    Fleet,
    /// The orchestration tree the GUI/TUI drives (unit structure, state, work, usage).
    Tree,
    /// One unit's node view.
    Unit {
        /// The unit id.
        id: String,
    },
    /// Drain recent management events for a unit (drill-down).
    UnitEvents {
        /// The unit id.
        id: String,
        /// Maximum events to drain (0 = all buffered).
        #[arg(long, default_value_t = 0)]
        max: u32,
    },
    /// Drain the rich §17 outbound stream for a unit (transcript-fidelity drill-down).
    UnitOutbound {
        /// The unit id.
        id: String,
        /// Maximum items to drain (0 = all buffered).
        #[arg(long, default_value_t = 0)]
        max: u32,
    },
    /// Pause a unit's scheduling (orchestrator sub-fleets).
    Pause {
        /// The unit id.
        id: String,
    },
    /// Resume a unit's scheduling.
    Resume {
        /// The unit id.
        id: String,
    },
    /// Scale a unit (sub-fleet) to N members.
    Scale {
        /// The unit id.
        id: String,
        /// The target member count.
        n: u32,
    },
    /// Open/continue an interactive session by starting a turn.
    Submit {
        /// The session id.
        id: String,
        /// The user message text.
        #[arg(default_value = "hello")]
        text: String,
    },
    /// Drain outbound events/requests from an interactive session.
    Poll {
        /// The session id.
        id: String,
        /// Maximum items to drain (0 = all available).
        #[arg(long, default_value_t = 0)]
        max: u32,
    },
    /// Read a session's durable verifiable history (non-destructive scroll-back).
    History {
        /// The session id.
        id: String,
        /// Return entries with cursor strictly greater than this (0 from the start).
        #[arg(long, default_value_t = 0)]
        after: u64,
        /// Maximum entries to return (0 = all available).
        #[arg(long, default_value_t = 0)]
        max: u32,
    },
    /// Read any unit's durable verifiable history (non-destructive scroll-back).
    UnitHistory {
        /// The unit id.
        id: String,
        /// Return entries with cursor strictly greater than this (0 from the start).
        #[arg(long, default_value_t = 0)]
        after: u64,
        /// Maximum entries to return (0 = all available).
        #[arg(long, default_value_t = 0)]
        max: u32,
    },
    /// Print the node's journal verifying key (hex dCBOR) for offline audit.
    VerifyingKey,
    /// Manage local-inference models (search/download/cache/catalog).
    Model {
        #[command(subcommand)]
        cmd: ModelCmd,
    },
    /// Curate a profile's skill library (usage view, pin/archive, run the deterministic curator).
    Curator {
        #[command(subcommand)]
        cmd: CuratorCmd,
    },
    /// Manage scheduled (cron) jobs and the consent-first suggestion catalog (I15).
    Cron {
        #[command(subcommand)]
        cmd: CronCmd,
    },
    /// List the registered transport adapters + their live instances (messaging-adapter framework).
    Transports,
    /// Manage conversations on a messaging transport (`room`, `matrix`, …). Aliased as `room`.
    #[command(alias = "room")]
    Conv {
        #[command(subcommand)]
        cmd: ConvCmd,
    },
    /// Administer membership of a conversation (invite/remove/ban/set-role).
    Member {
        #[command(subcommand)]
        cmd: MemberCmd,
    },
    /// Remote-contact operations on a messaging transport (`get-profile`, `set-alias`).
    Contact {
        #[command(subcommand)]
        cmd: ContactCmd,
    },
    /// Search a transport's contact/user directory.
    Directory {
        #[command(subcommand)]
        cmd: DirectoryCmd,
    },
}

/// Conversation management over the messaging-adapter interface (`conv_*` ops). `transport` is the
/// adapter family (`room`, `matrix`); `conv` is the adapter-opaque conversation id.
#[derive(Subcommand)]
pub(crate) enum ConvCmd {
    /// List conversations on a transport.
    List {
        /// The transport family (e.g. `room`).
        transport: String,
    },
    /// Show one conversation by id.
    Get {
        /// The transport family.
        transport: String,
        /// The conversation id.
        conv: String,
    },
    /// Create a conversation (Rooms reads `id`/`name`/`policy`/`kind` from these).
    Create {
        /// The transport family.
        transport: String,
        /// The conversation id (defaults to `name`, else generated).
        #[arg(long)]
        id: Option<String>,
        /// A human name/title.
        #[arg(long)]
        name: Option<String>,
        /// The floor policy (`addressed_only` | `free_for_all` | `round_robin`).
        #[arg(long)]
        policy: Option<String>,
        /// The conversation kind (`GroupDm` | `Channel` | `Dm` | `Thread`).
        #[arg(long)]
        kind: Option<String>,
    },
    /// Join (or create-if-absent) a channel by name.
    Join {
        /// The transport family.
        transport: String,
        /// The channel name / id.
        name: String,
    },
    /// Leave a conversation.
    Leave {
        /// The transport family.
        transport: String,
        /// The conversation id.
        conv: String,
    },
    /// Send a message into a conversation, optionally attributed to a member.
    Send {
        /// The transport family.
        transport: String,
        /// The conversation id.
        conv: String,
        /// The message text.
        text: String,
        /// Attribute the post to this member handle (an agent participant with `--from-profile`).
        #[arg(long)]
        from: Option<String>,
        /// The profile the `--from` member runs under (makes it an agent participant).
        #[arg(long)]
        from_profile: Option<String>,
    },
    /// Set a conversation's topic (omit to clear).
    Topic {
        /// The transport family.
        transport: String,
        /// The conversation id.
        conv: String,
        /// The new topic.
        topic: Option<String>,
    },
    /// Set a conversation's title (omit to clear).
    Title {
        /// The transport family.
        transport: String,
        /// The conversation id.
        conv: String,
        /// The new title.
        title: Option<String>,
    },
    /// Set a conversation's description (omit to clear).
    Describe {
        /// The transport family.
        transport: String,
        /// The conversation id.
        conv: String,
        /// The new description.
        description: Option<String>,
    },
    /// Delete/destroy a conversation.
    Delete {
        /// The transport family.
        transport: String,
        /// The conversation id.
        conv: String,
    },
    /// Read a conversation's durable, verifiable transcript (the merged room history).
    History {
        /// The transport family.
        transport: String,
        /// The conversation id.
        conv: String,
        /// Return entries with cursor strictly greater than this (0 from the start).
        #[arg(long, default_value_t = 0)]
        after: u64,
        /// Maximum entries (0 = all).
        #[arg(long, default_value_t = 0)]
        max: u32,
    },
}

/// Membership administration over the messaging-adapter interface (`member_*` ops). A `--profile`
/// makes the target an agent participant (`Participant::Agent`); otherwise it is a contact.
#[derive(Subcommand)]
pub(crate) enum MemberCmd {
    /// Invite/add a participant to a conversation.
    Invite {
        /// The transport family.
        transport: String,
        /// The conversation id.
        conv: String,
        /// The member handle.
        member: String,
        /// The profile the member runs under (agent participant).
        #[arg(long)]
        profile: Option<String>,
        /// An optional invitation message.
        #[arg(long)]
        message: Option<String>,
    },
    /// Remove/kick a participant.
    Remove {
        /// The transport family.
        transport: String,
        /// The conversation id.
        conv: String,
        /// The member handle.
        member: String,
        /// The profile the member runs under (agent participant).
        #[arg(long)]
        profile: Option<String>,
        /// An optional reason.
        #[arg(long)]
        reason: Option<String>,
    },
    /// Ban a participant.
    Ban {
        /// The transport family.
        transport: String,
        /// The conversation id.
        conv: String,
        /// The member handle.
        member: String,
        /// The profile the member runs under (agent participant).
        #[arg(long)]
        profile: Option<String>,
        /// An optional reason.
        #[arg(long)]
        reason: Option<String>,
    },
    /// Set a participant's role/affiliation.
    SetRole {
        /// The transport family.
        transport: String,
        /// The conversation id.
        conv: String,
        /// The member handle.
        member: String,
        /// The role (`none` | `voice` | `halfop` | `op` | `founder`).
        role: String,
        /// The profile the member runs under (agent participant).
        #[arg(long)]
        profile: Option<String>,
    },
}

/// Remote-contact operations over the messaging-adapter interface (`contact_*` ops). `transport` is
/// the instance id (`matrix/@user:hs`); `contact` is the contact id (a Matrix MXID `@user:hs`).
#[derive(Subcommand)]
pub(crate) enum ContactCmd {
    /// Fetch a remote contact's profile.
    GetProfile {
        /// The transport instance (`matrix/@user:hs`).
        transport: String,
        /// The contact id (MXID).
        contact: String,
    },
    /// Set a local alias for a contact (where the transport supports it).
    SetAlias {
        /// The transport instance.
        transport: String,
        /// The contact id (MXID).
        contact: String,
        /// The new alias (omit to clear).
        #[arg(long)]
        alias: Option<String>,
    },
}

/// Contact/user-directory search over the messaging-adapter interface (`directory_search`).
#[derive(Subcommand)]
pub(crate) enum DirectoryCmd {
    /// Search the directory for contacts/users.
    Search {
        /// The transport instance (`matrix/@user:hs`).
        transport: String,
        /// The search query (omit for an unfiltered listing where supported).
        #[arg(long)]
        query: Option<String>,
    },
}

/// Scheduled-job management: create/list/update/pause/resume/run-now/remove + recent runs, plus the
/// consent-first suggestion catalog (`suggest list|accept|dismiss`). Mirrors the operator `cron_*`
/// control ops; the same `CronOps` backs the agent `cron` tool.
#[derive(Subcommand)]
pub(crate) enum CronCmd {
    /// Create a scheduled job from a name + schedule (+ optional agent prompt).
    Create {
        /// The job name.
        name: String,
        /// The schedule: a cron expression (`"0 9 * * *"`), `@every <dur>`, or an ISO timestamp.
        schedule: String,
        /// The agent prompt the fired session runs (omit for an empty prompt).
        #[arg(long, default_value = "")]
        prompt: String,
        /// An IANA timezone for the schedule (defaults to the node's).
        #[arg(long)]
        timezone: Option<String>,
        /// Auto-delete after this many fires (omit for unlimited).
        #[arg(long)]
        repeat: Option<u32>,
        /// Create the job paused (disabled) instead of armed.
        #[arg(long)]
        disabled: bool,
    },
    /// List all scheduled jobs with their next fire time.
    List,
    /// Replace a job's schedule (+ optional prompt), recomputing its next fire.
    Update {
        /// The job id.
        id: String,
        /// The job name.
        name: String,
        /// The new schedule.
        schedule: String,
        /// The agent prompt (omit for an empty prompt).
        #[arg(long, default_value = "")]
        prompt: String,
    },
    /// Pause a job (clears its next fire).
    Pause {
        /// The job id.
        id: String,
    },
    /// Resume a paused job (recomputes its next fire from now).
    Resume {
        /// The job id.
        id: String,
    },
    /// Fire a job now (out of band), without advancing its schedule.
    Run {
        /// The job id.
        id: String,
    },
    /// Remove a job.
    Remove {
        /// The job id.
        id: String,
    },
    /// Show a job's recent runs.
    Runs {
        /// The job id.
        id: String,
    },
    /// Work with the consent-first suggestion catalog.
    Suggest {
        #[command(subcommand)]
        cmd: CronSuggestCmd,
    },
}

/// The suggestion-catalog verbs: list pending proposals, accept one (creates the job), or dismiss
/// one (latched by `dedup_key` so it is never re-offered).
#[derive(Subcommand)]
pub(crate) enum CronSuggestCmd {
    /// List pending suggestions (seeds the starter catalog on first use).
    List,
    /// Accept a suggestion: create its backing job (prints the new job id).
    Accept {
        /// The suggestion id.
        id: String,
    },
    /// Dismiss a suggestion (latched so it is never re-offered).
    Dismiss {
        /// The suggestion id.
        id: String,
    },
}

/// Per-profile skill curation: inspect usage/lifecycle, pin/archive by hand, or run the
/// deterministic curator (stale/archive/reactivate over the `.usage.json` sidecar).
#[derive(Subcommand)]
pub(crate) enum CuratorCmd {
    /// List a profile's skills with usage counts + lifecycle state.
    List {
        /// The profile id (defaults to the node's active default).
        #[arg(long)]
        profile: Option<String>,
    },
    /// Pin a skill (protect it from automatic archiving).
    Pin {
        /// The skill name.
        name: String,
        /// The profile id (defaults to the node's active default).
        #[arg(long)]
        profile: Option<String>,
    },
    /// Unpin a skill (re-expose it to automatic curation).
    Unpin {
        /// The skill name.
        name: String,
        /// The profile id (defaults to the node's active default).
        #[arg(long)]
        profile: Option<String>,
    },
    /// Archive a skill (move it out of discovery into `.archive/`).
    Archive {
        /// The skill name.
        name: String,
        /// The profile id (defaults to the node's active default).
        #[arg(long)]
        profile: Option<String>,
    },
    /// Restore an archived skill back into the live library.
    Restore {
        /// The skill name.
        name: String,
        /// The profile id (defaults to the node's active default).
        #[arg(long)]
        profile: Option<String>,
    },
    /// Run the deterministic curator (stale/archive/reactivate), printing the changes applied.
    Run {
        /// The profile id (defaults to the node's active default).
        #[arg(long)]
        profile: Option<String>,
    },
}

#[derive(Subcommand)]
pub(crate) enum ModelCmd {
    /// Search Hugging Face for models loadable by the given engine (step 1).
    Search {
        /// The free-text query.
        query: String,
        /// The target engine (llama|mistralrs).
        #[arg(long, default_value = "llama")]
        engine: String,
        /// Results per page.
        #[arg(long, default_value_t = 25)]
        limit: u32,
        /// The 0-based page.
        #[arg(long, default_value_t = 0)]
        page: u32,
    },
    /// List a repo's loadable files for the given engine (step 2).
    Files {
        /// The `org/name` repo id.
        repo: String,
        /// The git revision (defaults to `main`).
        #[arg(long)]
        revision: Option<String>,
        /// The target engine (llama|mistralrs).
        #[arg(long, default_value = "llama")]
        engine: String,
    },
    /// Download a model into the shared cache.
    Pull {
        /// The `org/name` repo id.
        repo: String,
        /// The GGUF file to fetch (required for llama; the first shard pulls the whole split set).
        #[arg(long)]
        file: Option<String>,
        /// The git revision (defaults to `main`).
        #[arg(long)]
        revision: Option<String>,
        /// The target engine (llama|mistralrs).
        #[arg(long, default_value = "llama")]
        engine: String,
    },
    /// List in-flight + finished download jobs.
    Downloads,
    /// Cancel a download job.
    Cancel {
        /// The download id (numeric).
        id: u64,
    },
    /// Pause a download job.
    Pause {
        /// The download id (numeric).
        id: u64,
    },
    /// Resume a paused/failed download job.
    Resume {
        /// The download id (numeric).
        id: u64,
    },
    /// List installed (cataloged) models.
    Ls,
    /// Delete an installed model (catalog record + cached artifact).
    Rm {
        /// The catalog id.
        id: String,
    },
    /// Activate a cataloged model so new worker spawns load it.
    Activate {
        /// The catalog id.
        id: String,
        /// The profile to activate it for (defaults to the node's default local profile).
        #[arg(long)]
        profile: Option<String>,
    },
    /// Recommend a quantization for a repo on the detected hardware.
    Recommend {
        /// The `org/name` repo id.
        repo: String,
        /// The target engine (llama|mistralrs).
        #[arg(long, default_value = "llama")]
        engine: String,
        /// The git revision (defaults to `main`).
        #[arg(long)]
        revision: Option<String>,
        /// Override the memory budget, in GiB (defaults to auto-detected VRAM/RAM).
        #[arg(long)]
        vram: Option<f64>,
    },
    /// Quantize a repo's GGUF to a target precision offline (via the llama-enabled worker).
    Quantize {
        /// The `org/name` repo id.
        repo: String,
        /// The target quant label (e.g. Q4_K_M).
        #[arg(long)]
        ftype: String,
        /// The source GGUF file (defaults to the highest-precision one in the repo).
        #[arg(long)]
        source: Option<String>,
        /// The git revision (defaults to `main`).
        #[arg(long)]
        revision: Option<String>,
    },
    /// List in-flight + finished quantization jobs.
    Quantizes,
    /// Inspect a cataloged model's GGUF metadata.
    Inspect {
        /// The catalog id.
        id: String,
    },
    /// Quickstart: recommend a quant, download it, and activate it — one command to get running.
    Up {
        /// The `org/name` repo id.
        repo: String,
        /// The target engine (llama|mistralrs).
        #[arg(long, default_value = "llama")]
        engine: String,
        /// Override the memory budget, in GiB (defaults to auto-detected VRAM/RAM).
        #[arg(long)]
        vram: Option<f64>,
        /// The profile to activate for (defaults to the node's default local profile).
        #[arg(long)]
        profile: Option<String>,
    },
}
