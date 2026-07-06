// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Linux in-process exec sandbox: Landlock (filesystem scope) + seccomp (network deny), installed
//! on the child via `pre_exec`. See the crate docs for the parent-builds / child-applies discipline.

use std::io;
use std::path::PathBuf;
use std::sync::OnceLock;

use landlock::{
    path_beneath_rules, Access, AccessFs, CompatLevel, Compatible, Ruleset, RulesetAttr,
    RulesetCreated, RulesetCreatedAttr, RulesetError, ABI,
};

use crate::SandboxSpec;

/// The Landlock ABI we require. V1 (Linux 5.13) covers the filesystem read/write/execute rights the
/// exec sandbox needs; network confinement is handled by seccomp, not Landlock, so no newer ABI is
/// required (maximizing kernel compatibility).
const ABI_LEVEL: ABI = ABI::V1;

/// Whether Landlock is usable on this host (cached once per process).
///
/// Probes with a *hard-requirement* ruleset create: if the running kernel's Landlock ABI is below
/// [`ABI_LEVEL`], `create()` errors and we report `false` — so the posture layer fails `Require`
/// closed rather than installing a silently-degraded ruleset.
pub(crate) fn available() -> bool {
    static CACHE: OnceLock<bool> = OnceLock::new();
    *CACHE.get_or_init(|| {
        Ruleset::default()
            .set_compatibility(CompatLevel::HardRequirement)
            .handle_access(AccessFs::from_all(ABI_LEVEL))
            .and_then(|r| r.create())
            .is_ok()
    })
}

/// Attach a `pre_exec` hook to `cmd` that installs Landlock + seccomp in the child.
///
/// The Landlock ruleset (which opens the allowed-path fds) and the seccomp BPF are built here, in
/// the **parent**. The closure moved into `pre_exec` runs after `fork`, before `execve`, in the
/// single-threaded child and performs **syscalls only** — `prctl(PR_SET_NO_NEW_PRIVS)`, a seccomp
/// apply (the kernel copies the pre-built program in — no child allocation), and Landlock
/// `restrict_self()` on the pre-built ruleset — so it does not allocate, lock, or touch
/// async-signal-unsafe state.
pub(crate) fn install(cmd: &mut tokio::process::Command, spec: &SandboxSpec) -> io::Result<()> {
    let ruleset = build_ruleset(spec).map_err(landlock_io)?;
    let bpf = build_seccomp(spec)?;

    // `restrict_self` consumes the ruleset; the `FnMut` closure holds it in an `Option` and takes it
    // on first (and only) invocation.
    let mut ruleset = Some(ruleset);

    // SAFETY: the closure runs post-`fork`, pre-`execve`, in the single-threaded child. It performs
    // only syscalls (`prctl`, `seccompiler::apply_filter` over an already-compiled program, and
    // `RulesetCreated::restrict_self` over an already-created ruleset) — no heap allocation, no lock
    // acquisition, no other async-signal-unsafe work — which is the discipline `pre_exec` requires.
    unsafe {
        cmd.pre_exec(move || {
            // NO_NEW_PRIVS is a precondition for an unprivileged seccomp filter and for Landlock.
            if libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0 {
                return Err(io::Error::last_os_error());
            }
            if let Some(bpf) = bpf.as_ref() {
                seccompiler::apply_filter(bpf)
                    .map_err(|e| io::Error::other(format!("seccomp apply failed: {e}")))?;
            }
            let ruleset = ruleset
                .take()
                .ok_or_else(|| io::Error::other("landlock ruleset already consumed"))?;
            ruleset
                .restrict_self()
                .map_err(|e| io::Error::other(format!("landlock restrict_self failed: {e}")))?;
            Ok(())
        });
    }
    Ok(())
}

/// Build the Landlock ruleset: default-deny once filesystem access is handled, granting read+execute
/// on the existing `ro_paths` and full access on the existing `rw_paths`. Non-existent paths are
/// skipped (a minimal/non-Nix host still works), mirroring bwrap's `--ro-bind-try`.
fn build_ruleset(spec: &SandboxSpec) -> Result<RulesetCreated, RulesetError> {
    let ro: Vec<PathBuf> = existing(&spec.ro_paths);
    let rw: Vec<PathBuf> = existing(&spec.rw_paths);
    // Built (fds opened) but NOT restricted here — `restrict_self` is called later in the child.
    Ruleset::default()
        .set_compatibility(CompatLevel::BestEffort)
        .handle_access(AccessFs::from_all(ABI_LEVEL))?
        .create()?
        .add_rules(path_beneath_rules(&ro, AccessFs::from_read(ABI_LEVEL)))?
        .add_rules(path_beneath_rules(&rw, AccessFs::from_all(ABI_LEVEL)))
}

/// The subset of `paths` that exist on this host.
fn existing(paths: &[PathBuf]) -> Vec<PathBuf> {
    paths.iter().filter(|p| p.exists()).cloned().collect()
}

/// Compile the seccomp filter denying `AF_INET`/`AF_INET6` sockets, or `None` when network is
/// allowed. A **targeted denylist** (default `Allow`, `socket(2)` matching those domains → `EACCES`)
/// — not a minimal-syscall allowlist, which would be fragile for arbitrary interpreters.
fn build_seccomp(spec: &SandboxSpec) -> io::Result<Option<seccompiler::BpfProgram>> {
    use seccompiler::{
        SeccompAction, SeccompCmpArgLen, SeccompCmpOp, SeccompCondition, SeccompFilter, SeccompRule,
    };
    use std::collections::BTreeMap;

    if spec.allow_network {
        return Ok(None);
    }

    let inet = |domain: libc::c_int| -> io::Result<SeccompRule> {
        SeccompRule::new(vec![SeccompCondition::new(
            0, // socket(2) arg 0 = domain
            SeccompCmpArgLen::Dword,
            SeccompCmpOp::Eq,
            domain as u64,
        )
        .map_err(seccomp_io)?])
        .map_err(seccomp_io)
    };

    let mut rules: BTreeMap<i64, Vec<SeccompRule>> = BTreeMap::new();
    rules.insert(
        libc::SYS_socket,
        vec![inet(libc::AF_INET)?, inet(libc::AF_INET6)?],
    );

    let filter = SeccompFilter::new(
        rules,
        SeccompAction::Allow, // default (no rule matches): allow
        SeccompAction::Errno(libc::EACCES as u32), // a matching socket() domain: EACCES
        std::env::consts::ARCH.try_into().map_err(seccomp_io)?,
    )
    .map_err(seccomp_io)?;

    let bpf: seccompiler::BpfProgram = filter.try_into().map_err(seccomp_io)?;
    Ok(Some(bpf))
}

fn landlock_io(e: RulesetError) -> io::Error {
    io::Error::other(format!("landlock ruleset build failed: {e}"))
}

fn seccomp_io<E: std::fmt::Display>(e: E) -> io::Error {
    io::Error::other(format!("seccomp filter build failed: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Stdio;

    /// The system paths a confined child needs read+execute on to start `/bin/sh` / `python3`
    /// (interpreter, dynamic linker, shared libs, certs). Non-existent entries are filtered by
    /// [`build_ruleset`], so this superset works on both NixOS and FHS hosts.
    fn system_ro() -> Vec<PathBuf> {
        [
            "/nix/store",
            "/usr",
            "/bin",
            "/lib",
            "/lib64",
            "/run/current-system/sw",
            "/etc",
            "/proc",
            "/dev",
            "/sys",
        ]
        .iter()
        .map(PathBuf::from)
        .collect()
    }

    fn tmp(tag: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("daemon-sandbox-{tag}-{nanos}"))
    }

    /// Run `/bin/sh -c <script>` with optional confinement; return `(success, stdout)`.
    async fn run_sh(spec: Option<&SandboxSpec>, script: &str) -> (bool, String) {
        let mut cmd = tokio::process::Command::new("/bin/sh");
        cmd.arg("-c")
            .arg(script)
            .env("PATH", std::env::var_os("PATH").unwrap_or_default())
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if let Some(spec) = spec {
            install(&mut cmd, spec).expect("install sandbox");
        }
        let out = cmd.output().await.expect("spawn");
        (
            out.status.success(),
            String::from_utf8_lossy(&out.stdout).into_owned(),
        )
    }

    fn python3() -> Option<String> {
        for name in ["python3", "python"] {
            let ok = std::process::Command::new(name)
                .arg("-c")
                .arg("import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)")
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .map(|s| s.success())
                .unwrap_or(false);
            if ok {
                return Some(name.to_string());
            }
        }
        None
    }

    // Bug reproduction: today's plain fallback runs unconfined, so a child can read any file the
    // daemon uid can. Under the Landlock backend a read outside the allowed scope is refused.
    #[tokio::test]
    async fn landlock_blocks_read_outside_scope() {
        if !available() {
            eprintln!("skipping landlock_blocks_read_outside_scope: Landlock unavailable");
            return;
        }
        let ws = tmp("read-ws");
        std::fs::create_dir_all(&ws).unwrap();
        std::fs::write(ws.join("allowed.txt"), b"in-scope").unwrap();
        let secret_dir = tmp("read-secret");
        std::fs::create_dir_all(&secret_dir).unwrap();
        let secret = secret_dir.join("secret.txt");
        std::fs::write(&secret, b"TOP SECRET").unwrap();

        let mut ro = system_ro();
        ro.push(ws.clone()); // read of the workspace is fine; the secret dir is deliberately absent.
        let spec = SandboxSpec {
            rw_paths: vec![ws.clone()],
            ro_paths: ro,
            allow_network: true, // isolate the fs assertion from the network filter
        };

        // Control: without the sandbox the same read succeeds (proves the bug the guard closes).
        let (ok_unconfined, out) = run_sh(None, &format!("cat {}", secret.display())).await;
        assert!(
            ok_unconfined && out.contains("TOP SECRET"),
            "control read should succeed"
        );

        // Confined: the out-of-scope read is refused.
        let (ok_confined, _) = run_sh(Some(&spec), &format!("cat {}", secret.display())).await;
        assert!(
            !ok_confined,
            "confined read of an out-of-scope file must fail"
        );

        // Confined: an in-scope read still works (we did not over-deny the interpreter/workspace).
        let (ok_inscope, inscope) = run_sh(
            Some(&spec),
            &format!("cat {}", ws.join("allowed.txt").display()),
        )
        .await;
        assert!(
            ok_inscope && inscope.contains("in-scope"),
            "in-scope read must succeed"
        );

        let _ = std::fs::remove_dir_all(&ws);
        let _ = std::fs::remove_dir_all(&secret_dir);
    }

    #[tokio::test]
    async fn landlock_blocks_write_outside_workspace() {
        if !available() {
            eprintln!("skipping landlock_blocks_write_outside_workspace: Landlock unavailable");
            return;
        }
        let ws = tmp("write-ws");
        std::fs::create_dir_all(&ws).unwrap();
        let outside_dir = tmp("write-outside");
        std::fs::create_dir_all(&outside_dir).unwrap();
        let outside = outside_dir.join("escape.txt");

        let mut ro = system_ro();
        ro.push(ws.clone());
        let spec = SandboxSpec {
            rw_paths: vec![ws.clone()],
            ro_paths: ro,
            allow_network: true,
        };

        let (ok, _) = run_sh(
            Some(&spec),
            &format!("echo escaped > {}", outside.display()),
        )
        .await;
        assert!(!ok, "confined write outside the workspace must fail");
        assert!(
            !outside.exists(),
            "the out-of-scope file must not be created"
        );

        // A write inside the rw workspace still works.
        let (ok_in, _) = run_sh(
            Some(&spec),
            &format!("echo ok > {}", ws.join("made.txt").display()),
        )
        .await;
        assert!(
            ok_in && ws.join("made.txt").exists(),
            "in-scope write must succeed"
        );

        let _ = std::fs::remove_dir_all(&ws);
        let _ = std::fs::remove_dir_all(&outside_dir);
    }

    // Bug reproduction: an unconfined child can open a network socket. With network denied the
    // seccomp filter refuses `AF_INET` socket creation (EACCES → Python OSError → non-zero exit).
    #[tokio::test]
    async fn seccomp_blocks_inet_socket() {
        if !available() {
            eprintln!("skipping seccomp_blocks_inet_socket: Landlock unavailable");
            return;
        }
        let Some(py) = python3() else {
            eprintln!("skipping seccomp_blocks_inet_socket: no usable python3");
            return;
        };
        let ws = tmp("net-ws");
        std::fs::create_dir_all(&ws).unwrap();
        let mut ro = system_ro();
        ro.push(ws.clone());

        let script = "import socket,sys\n\
                      try:\n    s=socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.close(); sys.exit(0)\n\
                      except OSError:\n    sys.exit(3)\n";

        let run = |allow_network: bool| {
            let py = py.clone();
            let ws = ws.clone();
            let ro = ro.clone();
            async move {
                let spec = SandboxSpec {
                    rw_paths: vec![ws.clone()],
                    ro_paths: ro,
                    allow_network,
                };
                let mut cmd = tokio::process::Command::new(&py);
                cmd.arg("-c")
                    .arg(script)
                    .current_dir(&ws)
                    .env("PATH", std::env::var_os("PATH").unwrap_or_default())
                    .stdin(Stdio::null())
                    .stdout(Stdio::null())
                    .stderr(Stdio::null());
                install(&mut cmd, &spec).expect("install");
                cmd.status().await.expect("spawn").success()
            }
        };

        // Network denied: the AF_INET socket() is refused.
        assert!(
            !run(false).await,
            "AF_INET socket must be denied when network is off"
        );
        // Network allowed: no seccomp filter, so socket creation succeeds.
        assert!(
            run(true).await,
            "AF_INET socket must succeed when network is allowed"
        );

        let _ = std::fs::remove_dir_all(&ws);
    }
}
