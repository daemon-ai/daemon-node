//! Edit-approval policy (§12 safety) — the per-session "session mode" governing whether a
//! mutating/dangerous tool action runs outright, is denied, or must be approved by a human.
//!
//! Ported from hermes' ACP session modes (`acp_adapter/server.py` `_session_modes`,
//! `acp_adapter/edit_approval.py`): a policy maps onto allow / deny / ask, with a conservative
//! carve-out so *sensitive* paths (`.git`/`.ssh`, dotenv files, private keys) always ask regardless
//! of the policy. The engine threads the effective policy onto each [`TurnCx`](crate::turn::TurnCx)
//! so a gated tool (fs edit, dangerous shell command) consults it before acting; the host decides
//! how an `Ask` is serviced (the live path parks for a human, the durable path suspends the turn
//! and resumes on the operator's decision).

use serde::{Deserialize, Serialize};
use std::path::Path;

/// The per-session edit-approval policy (the "session mode" a GUI selects). Mirrors hermes'
/// Default / Accept-Edits / Don't-Ask modes plus an explicit hard `Deny`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApprovalPolicy {
    /// Ask before every gated action (hermes "Default"). The host services the ask: the live path
    /// parks for a human, the durable path suspends the turn for an operator decision.
    #[default]
    Ask,
    /// Auto-allow workspace edits but still ask for *sensitive* paths (hermes "Accept Edits").
    AcceptEdits,
    /// Auto-allow every gated action except *sensitive* paths, which still ask (hermes "Don't Ask").
    AutoAllow,
    /// Deny every gated action outright (no prompt, no side effect).
    Deny,
}

/// The resolved decision for one gated action under a policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decision {
    /// Run the action without prompting.
    Allow,
    /// Reject the action (the tool returns a deny error; no side effect).
    Deny,
    /// Defer to the host: park for a human (live) or suspend the turn (durable).
    Ask,
}

impl ApprovalPolicy {
    /// The decision for a file-mutating action at `path`. Sensitive paths always ask (except under
    /// the hard `Deny`, which always denies).
    pub fn decide_edit(self, path: &str) -> Decision {
        match self {
            ApprovalPolicy::Deny => Decision::Deny,
            ApprovalPolicy::Ask => Decision::Ask,
            ApprovalPolicy::AcceptEdits | ApprovalPolicy::AutoAllow => {
                if is_sensitive_path(path) {
                    Decision::Ask
                } else {
                    Decision::Allow
                }
            }
        }
    }

    /// The decision for a non-path gated action (e.g. a dangerous shell command). `AcceptEdits` is
    /// about file edits, so it still asks for commands; `AutoAllow` allows; `Deny` denies.
    pub fn decide_command(self) -> Decision {
        match self {
            ApprovalPolicy::Deny => Decision::Deny,
            ApprovalPolicy::Ask | ApprovalPolicy::AcceptEdits => Decision::Ask,
            ApprovalPolicy::AutoAllow => Decision::Allow,
        }
    }
}

/// File names that always require approval regardless of policy (secrets / credentials).
const SENSITIVE_NAMES: &[&str] = &[
    ".env",
    ".env.local",
    ".env.production",
    "id_rsa",
    "id_ed25519",
];

/// Whether a path is *sensitive* and must always ask: anything under a `.git`/`.ssh` directory, or
/// a dotenv / private-key file by name (ported from hermes `_is_sensitive_auto_approve_path`).
pub fn is_sensitive_path(path: &str) -> bool {
    let p = Path::new(path);
    for component in p.components() {
        let part = component.as_os_str().to_string_lossy().to_lowercase();
        if part == ".git" || part == ".ssh" {
            return true;
        }
    }
    match p.file_name() {
        Some(name) => {
            let lower = name.to_string_lossy().to_lowercase();
            SENSITIVE_NAMES.contains(&lower.as_str())
        }
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sensitive_paths_detected() {
        assert!(is_sensitive_path("/home/u/project/.git/config"));
        assert!(is_sensitive_path("/home/u/.ssh/known_hosts"));
        assert!(is_sensitive_path("/srv/app/.env"));
        assert!(is_sensitive_path("/srv/app/.env.production"));
        assert!(is_sensitive_path("/home/u/.ssh/id_ed25519"));
        assert!(!is_sensitive_path("/srv/app/src/main.rs"));
        assert!(!is_sensitive_path("notes.md"));
    }

    #[test]
    fn policy_decisions() {
        // Ask asks for everything; Deny denies everything.
        assert_eq!(ApprovalPolicy::Ask.decide_edit("a.txt"), Decision::Ask);
        assert_eq!(ApprovalPolicy::Deny.decide_edit("a.txt"), Decision::Deny);
        // Auto-allow runs ordinary edits but still asks for sensitive ones.
        assert_eq!(
            ApprovalPolicy::AutoAllow.decide_edit("a.txt"),
            Decision::Allow
        );
        assert_eq!(
            ApprovalPolicy::AutoAllow.decide_edit("/app/.env"),
            Decision::Ask
        );
        // Accept-edits allows ordinary edits but asks for commands.
        assert_eq!(
            ApprovalPolicy::AcceptEdits.decide_edit("a.txt"),
            Decision::Allow
        );
        assert_eq!(ApprovalPolicy::AcceptEdits.decide_command(), Decision::Ask);
        assert_eq!(ApprovalPolicy::AutoAllow.decide_command(), Decision::Allow);
    }

    #[test]
    fn default_is_ask() {
        assert_eq!(ApprovalPolicy::default(), ApprovalPolicy::Ask);
    }
}
