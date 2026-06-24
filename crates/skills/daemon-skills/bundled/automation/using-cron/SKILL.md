---
name: using-cron
description: "How to schedule recurring work with the cron tool: actions, fields, safety."
version: 1.0.0
author: Daemon
license: MIT
platforms: [linux, macos, windows]
metadata:
  daemon:
    tags: [automation, cron, scheduling, recurring, reminders, blueprint]
    related_skills: [dependency-audit]
    requires_tools: [cron]
---

# Using the cron tool

Use the `cron` tool to **schedule recurring or future work**: briefings, monitors,
reminders, digests, and chained data pipelines. A scheduled run fires in its **own
isolated session with no chat memory**, so every job must be self-contained.

Reach for this when the user says things like "every morning…", "remind me to…",
"check X every 30 minutes", "each Friday…", or "set up an automation".

## Actions

The tool takes an `action`:

| Action | Required | What it does |
|--------|----------|--------------|
| `create` | `name`, `schedule` (+ `prompt` unless `no_agent`) | Schedule a new job |
| `list` | — | List jobs with next-fire and last-run status |
| `update` | `id` (+ fields) | Edit an existing job |
| `pause` / `resume` | `id` | Disarm / re-arm without deleting |
| `run` | `id` | Fire once now ("run now") |
| `remove` | `id` | Delete the job |

## Schedule grammar

`schedule` accepts three forms (parsed by the node):

- **Cron expression** — `"0 9 * * *"` (09:00 daily), `"*/15 * * * *"`, `"0 18 * * 0"`
  (Sundays 18:00), `"0 9 * * 1-5"` (weekdays 09:00). Set `timezone` (IANA, e.g.
  `"Europe/Berlin"`) when the wall-clock time matters.
- **Interval** — `"@every 30m"`, `"@every 2h"`.
- **One-shot** — a single ISO-8601 timestamp.

## Fields you can set

- `prompt` — the instruction the run executes. **Be self-contained** (no chat memory).
- `repeat` — auto-delete after N fires; omit for unlimited.
- `jitter_secs` — spread herds of identically-scheduled jobs.
- `overlap` — `skip` (default), `allow`, or `queue` when a fire overlaps a running one.
- `catch_up` — `grace` (default), `skip`, or `always` for missed fires after downtime.
- `deliver` — `"origin"` (back to this chat), `"all"`, `"<transport>:<chat>"`, or omit
  for store-only. **Omit `deliver` for silent/internal jobs.**
- `skills` — skill names to **preload** into the run (their content is injected ahead
  of the prompt), so a job carries the same skill context a chat would.
- `context_from` — job ids whose latest output is injected first (chain jobs: A collects,
  B processes).
- `script` + `no_agent` — run a node-scripts-relative script only, no LLM turn (its stdout
  is delivered verbatim; empty stdout = silent).
- `enabled_toolsets`, `workdir`, `model`, `provider` — per-job run shaping.

## The `[SILENT]` convention

For monitors and digests, instruct the run to reply with **exactly `[SILENT]`** when
there is nothing worth delivering. A noisy job that reports "nothing new" every tick
trains the user to ignore it.

## Examples

Daily briefing, delivered back to this chat:

```
cron(action="create", name="Daily briefing", schedule="0 8 * * *",
     deliver="origin",
     prompt="Give a concise briefing: today's calendar, weather, and anything urgent.")
```

Inbox monitor every 30 minutes, silent when quiet:

```
cron(action="create", name="Inbox monitor", schedule="@every 30m", deliver="origin",
     prompt="Surface only genuinely urgent inbox items. If nothing is urgent, reply exactly [SILENT].")
```

A two-stage pipeline (collector → summarizer):

```
cron(action="create", name="Collect metrics", schedule="0 * * * *", script="scripts/collect.sh", no_agent=true)
cron(action="create", name="Hourly summary", schedule="5 * * * *", context_from=["<collector-id>"],
     prompt="Summarize the latest collected metrics for the user.")
```

## Suggestions & blueprints

Don't free-hand a raw cron expression when a **blueprint** fits — blueprints collect a
time/weekday/choice and fill a vetted schedule for you. Skills can also *be* blueprints
(a `metadata.daemon.blueprint` block), which appear as **consent-first suggestions** the
user accepts; accepting one schedules a job that preloads that skill.

## Safety

- A scheduled run **cannot create or manage cron jobs** — the `cron` tool is absent
  inside cron-fired sessions. Don't try to self-schedule from within a job.
- Scripts are sandboxed under the node's scripts directory: relative paths only, no `..`.
- Always confirm a schedule with the user before creating recurring jobs on their behalf.
