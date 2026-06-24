---
name: dependency-audit
description: "Weekly dependency & advisory audit; runnable as a scheduled automation."
version: 1.0.0
author: Daemon (cron blueprint example)
license: MIT
platforms: [linux, macos, windows]
metadata:
  daemon:
    tags: [automation, dependencies, security, audit, blueprint, cron]
    related_skills: [using-cron]
    blueprint:
      schedule: "0 9 * * 1"
      deliver: origin
      prompt: "Run the weekly dependency audit described in this skill and report only what changed or needs attention. If nothing is actionable, respond with exactly \"[SILENT]\"."
---

# Dependency audit

Use this skill to review a project's dependencies for **outdated versions, known
advisories, and drift from the lockfile** — and to keep that review on a schedule.

This skill is also a **blueprint**: its `metadata.daemon.blueprint` block declares a
weekly schedule (`0 9 * * 1` — Mondays at 09:00), so installing it offers a
consent-first *cron suggestion*. Accepting the suggestion creates a cron job that
**preloads this skill** and runs the prompt above. It never schedules itself — you
accept it via the suggestions surface (or the `cron` tool's `run`/`create`).

## When to use

- The user asks to "check dependencies", "audit packages", "what's out of date?",
  or wants a recurring health check on a project's supply chain.

## Method

1. **Detect the ecosystem.** Look for `Cargo.toml`/`Cargo.lock`, `package.json`/lockfiles,
   `pyproject.toml`/`requirements.txt`, `go.mod`, etc. Audit each present ecosystem.
2. **List outdated.** Use the ecosystem's own tool where available
   (`cargo outdated`, `npm outdated`, `pip list --outdated`, `go list -m -u all`).
3. **Check advisories.** Run the audit tool if present (`cargo audit`, `npm audit`,
   `pip-audit`, `govulncheck`). Summarize severity, not raw output.
4. **Report deltas only.** When run as a scheduled job, report what changed since the
   last run and what needs attention. If nothing is actionable, emit exactly `[SILENT]`
   so the run delivers nothing.

## Output

A short, skimmable report grouped by ecosystem:

```markdown
## Dependency audit — <date>

### Rust (Cargo)
- `tokio` 1.38 → 1.40 (minor; changelog: ...)
- advisory: RUSTSEC-2025-XXXX in `foo` (medium) — upgrade to >= 1.2

### Recommended actions
- [ ] Bump `tokio`, run the test suite
- [ ] Patch `foo` for the advisory
```

Keep it actionable. A scheduled run that says "everything is fine" every week trains
the user to ignore it — prefer `[SILENT]` when there is genuinely nothing to do.
