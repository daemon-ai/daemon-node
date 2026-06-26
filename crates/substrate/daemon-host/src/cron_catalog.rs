//! Cron suggestion catalog + parameterized blueprints (I15 / Workstream H; v16 skill bridge).
//!
//! Consent-first scheduling UX, data-only so it is trivially extendable. Three layers, all compiling
//! down to the one [`CronSpec`] job engine:
//!
//! - **Starter catalog** ([`starter_suggestions`]): a small set of ready-to-accept
//!   [`CronSuggestion`]s (Hermes' four starters, ported + aligned) the node seeds as `Pending` on
//!   first run. Accepting one calls `cron_create(spec)`; dismissing latches it by `dedup_key` so it
//!   is never re-offered (the latch lives in [`CronOps`](crate::cron::CronOps)).
//! - **Blueprints** ([`blueprints`]): parameterized templates (Hermes' 14, ported) with typed
//!   [`SlotKind`]s so a GUI/CLI can collect a time / weekday / choice / text and
//!   [`CronBlueprint::fill`] it into a concrete [`CronSpec`] — the user never types a raw cron
//!   expression. Filling is pure token substitution, so a blueprint is just data.
//! - **Skill bridge** ([`blueprint_suggestion`]): a skill whose `metadata.daemon.blueprint` declares
//!   a schedule becomes a `Pending` suggestion (source `"blueprint"`) the moment it is installed —
//!   accept compiles it into a job that preloads that skill (it never auto-schedules).

use daemon_api::{CronSpec, CronSuggestion, SuggestionStatus};
use daemon_skills::SkillBlueprint;
use std::collections::BTreeMap;

/// A blueprint slot's input type, so a GUI renders the right control (and the CLI validates).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlotKind {
    /// A `HH:MM` 24-hour wall-clock time. Expands to `{minute}` + `{hour}` schedule tokens.
    Time,
    /// A comma-separated weekday set (`mon,tue,...` or `*`). Expands to the `{weekdays}` token.
    Weekdays,
    /// One of a fixed set of choices.
    Choice(&'static [&'static str]),
    /// Free text (e.g. a prompt detail). Expands to its `{key}` token verbatim.
    Text,
}

/// One typed input a blueprint collects before it can be filled into a [`CronSpec`].
#[derive(Clone, Copy, Debug)]
pub struct BlueprintSlot {
    /// The token key (substituted as `{key}` in the templates). The reserved key `"deliver"` sets
    /// the job's delivery routing rather than a template token.
    pub key: &'static str,
    /// A human label for the input.
    pub label: &'static str,
    /// The input's type.
    pub kind: SlotKind,
    /// The default value used when the caller does not supply one.
    pub default: &'static str,
}

/// A parameterized cron-job template (data-only): typed slots + templated name/schedule/prompt that
/// [`fill`](CronBlueprint::fill) turns into a concrete [`CronSpec`].
#[derive(Clone, Copy, Debug)]
pub struct CronBlueprint {
    /// The stable blueprint id.
    pub id: &'static str,
    /// A short title.
    pub title: &'static str,
    /// What the produced job does.
    pub description: &'static str,
    /// A coarse grouping for GUI presentation (`daily`/`weekly`/`monitor`/`wellbeing`/...).
    pub category: &'static str,
    /// Search/curation tags.
    pub tags: &'static [&'static str],
    /// The job-name template (may contain `{key}` tokens).
    pub name: &'static str,
    /// The schedule template. May contain `{minute}`/`{hour}` (from a `Time` slot) and `{weekdays}`
    /// (from a `Weekdays` slot), plus any `{key}` text/choice tokens.
    pub schedule: &'static str,
    /// The agent prompt template (may contain `{key}` tokens).
    pub prompt: &'static str,
    /// The default delivery routing (`"origin"` delivers back to the originating chat; `"local"`/
    /// empty = store-only). A `"deliver"` slot overrides this per-fill.
    pub deliver_default: &'static str,
    /// Skill names the produced job preloads (v16 `CronSpec::skills`).
    pub skills: &'static [&'static str],
    /// The typed inputs this blueprint collects.
    pub slots: &'static [BlueprintSlot],
}

impl CronBlueprint {
    /// Fill this blueprint into a concrete [`CronSpec`] from `values` (missing slots use their
    /// [`default`](BlueprintSlot::default)). A `Time` slot (`HH:MM`) expands to `{minute}`/`{hour}`;
    /// a `Weekdays` slot to `{weekdays}`; a `"deliver"` slot sets the job's routing; other slots
    /// substitute `{key}` verbatim. Unparseable time falls back to the slot default.
    pub fn fill(&self, values: &BTreeMap<String, String>) -> CronSpec {
        let mut tokens: BTreeMap<String, String> = BTreeMap::new();
        let mut deliver_value: Option<String> = None;
        for slot in self.slots {
            let raw = values
                .get(slot.key)
                .map(String::as_str)
                .unwrap_or(slot.default);
            match slot.kind {
                SlotKind::Time => {
                    let (minute, hour) = parse_hhmm(raw)
                        .unwrap_or_else(|| parse_hhmm(slot.default).unwrap_or((0, 9)));
                    tokens.insert("minute".into(), minute.to_string());
                    tokens.insert("hour".into(), hour.to_string());
                }
                SlotKind::Weekdays => {
                    tokens.insert("weekdays".into(), normalize_weekdays(raw));
                }
                SlotKind::Choice(_) | SlotKind::Text => {
                    if slot.key == "deliver" {
                        deliver_value = Some(raw.to_string());
                    }
                    tokens.insert(slot.key.into(), raw.to_string());
                }
            }
        }
        let deliver = deliver_value.unwrap_or_else(|| self.deliver_default.to_string());
        CronSpec {
            name: substitute(self.name, &tokens),
            schedule: substitute(self.schedule, &tokens),
            payload: substitute(self.prompt, &tokens).into_bytes(),
            enabled: true,
            deliver: resolve_deliver(&deliver),
            skills: self.skills.iter().map(|s| s.to_string()).collect(),
            ..CronSpec::default()
        }
    }
}

/// Resolve a delivery-routing string into the wire `Option<String>`: `"local"`/`"none"`/empty means
/// store-only (`None`); anything else (e.g. `"origin"`, `"all"`, `"<transport>:<chat>"`) is kept.
fn resolve_deliver(s: &str) -> Option<String> {
    let t = s.trim();
    if t.is_empty() || t.eq_ignore_ascii_case("local") || t.eq_ignore_ascii_case("none") {
        None
    } else {
        Some(t.to_string())
    }
}

/// Parse `HH:MM` into `(minute, hour)`; `None` if malformed or out of range.
fn parse_hhmm(s: &str) -> Option<(u32, u32)> {
    let (h, m) = s.trim().split_once(':')?;
    let hour: u32 = h.trim().parse().ok()?;
    let minute: u32 = m.trim().parse().ok()?;
    (hour < 24 && minute < 60).then_some((minute, hour))
}

/// Normalize a weekday set: `*`/empty -> `*`, else the trimmed lowercase comma list.
fn normalize_weekdays(s: &str) -> String {
    let t = s.trim();
    if t.is_empty() || t == "*" {
        return "*".into();
    }
    t.split(',')
        .map(|d| d.trim().to_lowercase())
        .filter(|d| !d.is_empty())
        .collect::<Vec<_>>()
        .join(",")
}

/// Substitute `{key}` tokens in `template` from `tokens` (unknown tokens are left intact).
fn substitute(template: &str, tokens: &BTreeMap<String, String>) -> String {
    let mut out = template.to_string();
    for (key, value) in tokens {
        out = out.replace(&format!("{{{key}}}"), value);
    }
    out
}

// --- reusable slot building blocks ------------------------------------------------------------

const SLOT_DELIVER: BlueprintSlot = BlueprintSlot {
    key: "deliver",
    label: "Deliver to",
    kind: SlotKind::Choice(&["origin", "all", "local"]),
    default: "origin",
};

/// A `Time` slot with the given default `HH:MM` (const-friendly factory).
const fn time_slot(default: &'static str) -> BlueprintSlot {
    BlueprintSlot {
        key: "time",
        label: "Time of day",
        kind: SlotKind::Time,
        default,
    }
}

const fn weekdays_slot(default: &'static str) -> BlueprintSlot {
    BlueprintSlot {
        key: "weekdays",
        label: "Days of week",
        kind: SlotKind::Weekdays,
        default,
    }
}

const fn text_slot(key: &'static str, label: &'static str, default: &'static str) -> BlueprintSlot {
    BlueprintSlot {
        key,
        label,
        kind: SlotKind::Text,
        default,
    }
}

const INTERVAL_SLOT: BlueprintSlot = BlueprintSlot {
    key: "interval",
    label: "Check every (minutes)",
    kind: SlotKind::Choice(&["15", "30", "60"]),
    default: "30",
};

/// The built-in blueprints — Hermes' 14-template catalog, ported and adapted to the daemon
/// schedule grammar (data-only; extend by appending entries).
pub fn blueprints() -> &'static [CronBlueprint] {
    const MORNING_BRIEF: &[BlueprintSlot] = &[time_slot("08:00"), SLOT_DELIVER];
    const IMPORTANT_MAIL: &[BlueprintSlot] = &[INTERVAL_SLOT, SLOT_DELIVER];
    const WEEKLY_REVIEW: &[BlueprintSlot] =
        &[weekdays_slot("sun"), time_slot("18:00"), SLOT_DELIVER];
    const WORKDAY_START: &[BlueprintSlot] = &[time_slot("09:00"), SLOT_DELIVER];
    const CUSTOM_REMINDER: &[BlueprintSlot] = &[
        time_slot("10:00"),
        text_slot("what", "What to remind about", "take a short break"),
        SLOT_DELIVER,
    ];
    const EVENING_WINDDOWN: &[BlueprintSlot] = &[time_slot("21:00"), SLOT_DELIVER];
    const NEWS_DIGEST: &[BlueprintSlot] = &[
        time_slot("18:00"),
        text_slot("topic", "Topic", "technology"),
        SLOT_DELIVER,
    ];
    const BILL_WATCH: &[BlueprintSlot] = &[weekdays_slot("mon"), time_slot("09:00"), SLOT_DELIVER];
    const HABIT_CHECKIN: &[BlueprintSlot] = &[
        time_slot("20:00"),
        text_slot("habit", "Habit", "your daily habit"),
        SLOT_DELIVER,
    ];
    const HYDRATION_MOVE: &[BlueprintSlot] = &[SLOT_DELIVER];
    const MEAL_PLAN: &[BlueprintSlot] = &[weekdays_slot("sun"), time_slot("10:00"), SLOT_DELIVER];
    const LEARN_DAILY: &[BlueprintSlot] = &[
        time_slot("12:00"),
        text_slot("topic", "Topic", "something new"),
        SLOT_DELIVER,
    ];
    const GRATITUDE: &[BlueprintSlot] = &[time_slot("21:00"), SLOT_DELIVER];
    const ON_THIS_DAY: &[BlueprintSlot] = &[time_slot("08:00"), SLOT_DELIVER];
    &[
        CronBlueprint {
            id: "morning-brief",
            title: "Morning briefing",
            description: "A short daily briefing: today's calendar, weather, and anything urgent.",
            category: "daily",
            tags: &["daily", "briefing"],
            name: "Morning briefing",
            schedule: "{minute} {hour} * * *",
            prompt: "Produce a concise morning briefing for the user: today's calendar events, the \
                     local weather, and any urgent items waiting on them. Keep it short and skimmable.",
            deliver_default: "origin",
            skills: &[],
            slots: MORNING_BRIEF,
        },
        CronBlueprint {
            id: "important-mail",
            title: "Important-mail monitor",
            description: "Periodically scan the inbox and surface only genuinely urgent messages.",
            category: "monitor",
            tags: &["mail", "monitor"],
            name: "Important-mail monitor",
            schedule: "@every {interval}m",
            prompt: "Check the inbox for genuinely urgent or time-sensitive messages and summarize \
                     them. If there is nothing urgent, respond with exactly \"[SILENT]\" and nothing \
                     else to suppress delivery.",
            deliver_default: "origin",
            skills: &[],
            slots: IMPORTANT_MAIL,
        },
        CronBlueprint {
            id: "weekly-review",
            title: "Weekly review",
            description: "A weekly recap and a look ahead to next week's priorities.",
            category: "weekly",
            tags: &["weekly", "review"],
            name: "Weekly review",
            schedule: "{minute} {hour} * * {weekdays}",
            prompt: "Review what happened this week and outline the priorities for next week. \
                     Be concrete and concise.",
            deliver_default: "origin",
            skills: &[],
            slots: WEEKLY_REVIEW,
        },
        CronBlueprint {
            id: "workday-start",
            title: "Workday start reminder",
            description: "A weekday nudge of the day's top priorities.",
            category: "daily",
            tags: &["work", "reminder"],
            name: "Workday start reminder",
            schedule: "{minute} {hour} * * 1-5",
            prompt: "Remind the user of their top priorities to start the workday. Keep it to a few \
                     bullet points.",
            deliver_default: "origin",
            skills: &[],
            slots: WORKDAY_START,
        },
        CronBlueprint {
            id: "custom-reminder",
            title: "Custom reminder",
            description: "A simple recurring reminder of your choosing.",
            category: "daily",
            tags: &["reminder"],
            name: "Reminder",
            schedule: "{minute} {hour} * * *",
            prompt: "Reminder: {what}.",
            deliver_default: "origin",
            skills: &[],
            slots: CUSTOM_REMINDER,
        },
        CronBlueprint {
            id: "evening-winddown",
            title: "Evening wind-down",
            description: "An end-of-day prompt to wrap up and prep for tomorrow.",
            category: "daily",
            tags: &["evening", "wellbeing"],
            name: "Evening wind-down",
            schedule: "{minute} {hour} * * *",
            prompt: "Help the user wind down: a brief recap of what they got done today and a short \
                     prep list for tomorrow.",
            deliver_default: "origin",
            skills: &[],
            slots: EVENING_WINDDOWN,
        },
        CronBlueprint {
            id: "news-digest",
            title: "Topic news digest",
            description: "A recurring digest on a topic; silent when there's nothing new.",
            category: "monitor",
            tags: &["news", "digest"],
            name: "News digest: {topic}",
            schedule: "{minute} {hour} * * *",
            prompt: "Produce a short digest of genuinely new developments about \"{topic}\" since \
                     the last run. If there is nothing new, respond with exactly \"[SILENT]\".",
            deliver_default: "origin",
            skills: &[],
            slots: NEWS_DIGEST,
        },
        CronBlueprint {
            id: "bill-renewal-watch",
            title: "Bills & renewals reminder",
            description: "A recurring reminder to review upcoming bills and subscription renewals.",
            category: "weekly",
            tags: &["finance", "reminder"],
            name: "Bills & renewals",
            schedule: "{minute} {hour} * * {weekdays}",
            prompt: "Remind the user of upcoming bills and subscription renewals to review. Flag \
                     anything that looks unusual.",
            deliver_default: "origin",
            skills: &[],
            slots: BILL_WATCH,
        },
        CronBlueprint {
            id: "habit-checkin",
            title: "Habit check-in",
            description: "A daily nudge to check in on a habit you're building.",
            category: "wellbeing",
            tags: &["habit", "wellbeing"],
            name: "Habit check-in: {habit}",
            schedule: "{minute} {hour} * * *",
            prompt: "Check in with the user about their habit: \"{habit}\". Ask how it went today \
                     and offer one small encouragement.",
            deliver_default: "origin",
            skills: &[],
            slots: HABIT_CHECKIN,
        },
        CronBlueprint {
            id: "hydration-move",
            title: "Hydration & movement nudge",
            description: "Hourly weekday nudges (9am-5pm) to drink water and move.",
            category: "wellbeing",
            tags: &["health", "wellbeing"],
            name: "Hydration & movement",
            schedule: "0 9-17 * * 1-5",
            prompt: "Send a brief, friendly nudge to drink some water and move for a minute.",
            deliver_default: "origin",
            skills: &[],
            slots: HYDRATION_MOVE,
        },
        CronBlueprint {
            id: "meal-plan",
            title: "Weekly meal plan",
            description: "A weekly prompt to plan meals and build a shopping list.",
            category: "weekly",
            tags: &["food", "planning"],
            name: "Weekly meal plan",
            schedule: "{minute} {hour} * * {weekdays}",
            prompt: "Help the user plan meals for the coming week and draft a shopping list.",
            deliver_default: "origin",
            skills: &[],
            slots: MEAL_PLAN,
        },
        CronBlueprint {
            id: "learn-daily",
            title: "Daily learning drip",
            description: "A weekday micro-lesson on a topic of your choosing.",
            category: "daily",
            tags: &["learning"],
            name: "Daily learning: {topic}",
            schedule: "{minute} {hour} * * 1-5",
            prompt: "Teach the user one bite-sized lesson about \"{topic}\". Keep it to a couple of \
                     paragraphs with one concrete takeaway.",
            deliver_default: "origin",
            skills: &[],
            slots: LEARN_DAILY,
        },
        CronBlueprint {
            id: "gratitude-journal",
            title: "Gratitude & reflection prompt",
            description: "A daily reflective prompt for journaling.",
            category: "wellbeing",
            tags: &["journal", "wellbeing"],
            name: "Gratitude prompt",
            schedule: "{minute} {hour} * * *",
            prompt: "Offer the user a short, thoughtful gratitude or reflection prompt for today.",
            deliver_default: "origin",
            skills: &[],
            slots: GRATITUDE,
        },
        CronBlueprint {
            id: "on-this-day",
            title: "On-this-day discovery",
            description: "A daily snippet of something interesting that happened on this date.",
            category: "daily",
            tags: &["discovery", "fun"],
            name: "On this day",
            schedule: "{minute} {hour} * * *",
            prompt: "Share one genuinely interesting thing that happened on this date in history. \
                     Keep it to a short paragraph.",
            deliver_default: "origin",
            skills: &[],
            slots: ON_THIS_DAY,
        },
    ]
}

/// The starter suggestions the node offers on first run (Hermes' four starters, ported + aligned).
/// Each is a concrete, ready-to-accept [`CronSuggestion`] with a stable `dedup_key` so accept/dismiss
/// latches. Starters deliver back to the originating chat (`deliver = origin`).
pub fn starter_suggestions() -> Vec<CronSuggestion> {
    fn agent_spec(name: &str, schedule: &str, prompt: &str) -> CronSpec {
        CronSpec {
            name: name.into(),
            schedule: schedule.into(),
            payload: prompt.as_bytes().to_vec(),
            enabled: true,
            deliver: Some("origin".into()),
            ..CronSpec::default()
        }
    }
    fn suggestion(key: &str, title: &str, description: &str, spec: CronSpec) -> CronSuggestion {
        CronSuggestion {
            id: format!("starter-{key}"),
            title: title.into(),
            description: description.into(),
            source: "catalog".into(),
            spec,
            dedup_key: format!("catalog:{key}"),
            status: SuggestionStatus::Pending,
        }
    }
    vec![
        suggestion(
            "daily-briefing",
            "Daily briefing",
            "Every morning at 8am: today's calendar, weather, and anything urgent waiting on you.",
            agent_spec(
                "Daily briefing",
                "0 8 * * *",
                "Produce a concise morning briefing for the user: today's calendar events, the \
                 local weather, and any urgent items.",
            ),
        ),
        suggestion(
            "important-mail-monitor",
            "Important-mail monitor",
            "Every 30 minutes, surface only genuinely urgent inbox messages (silent otherwise).",
            agent_spec(
                "Important-mail monitor",
                "@every 30m",
                "Check the inbox for genuinely urgent messages and summarize them. If there is \
                 nothing urgent, respond with exactly \"[SILENT]\".",
            ),
        ),
        suggestion(
            "weekly-review",
            "Weekly review",
            "Every Sunday evening, recap the week and look ahead to the next.",
            agent_spec(
                "Weekly review",
                "0 18 * * 0",
                "Review what happened this week and outline priorities for next week.",
            ),
        ),
        suggestion(
            "workday-start",
            "Workday start reminder",
            "Weekdays at 9am, a nudge of your top priorities for the day.",
            agent_spec(
                "Workday start reminder",
                "0 9 * * 1-5",
                "Remind me of my top priorities to start the workday.",
            ),
        ),
    ]
}

/// Bridge a skill's `metadata.daemon.blueprint` automation into a consent-first [`CronSuggestion`]
/// (source `"blueprint"`). The produced job preloads `skill_name` (v16 `CronSpec::skills`) and is
/// latched by `dedup_key = "blueprint:{skill}:{schedule}"`, so re-scanning never duplicates an
/// already-acted suggestion. Returns `None` if the block declares no schedule.
pub fn blueprint_suggestion(skill_name: &str, bp: &SkillBlueprint) -> Option<CronSuggestion> {
    if !bp.is_runnable() {
        return None;
    }
    let schedule = bp.schedule.trim().to_string();
    let spec = CronSpec {
        name: format!("blueprint:{skill_name}"),
        schedule: schedule.clone(),
        payload: bp.prompt.clone().unwrap_or_default().into_bytes(),
        enabled: true,
        no_agent: bp.no_agent,
        deliver: resolve_deliver(bp.deliver.as_deref().unwrap_or("origin")),
        model: bp.model.clone(),
        provider: bp.provider.clone(),
        enabled_toolsets: bp.enabled_toolsets.clone(),
        skills: vec![skill_name.to_string()],
        ..CronSpec::default()
    };
    Some(CronSuggestion {
        id: format!("blueprint-{skill_name}"),
        title: format!("Automation: {skill_name}"),
        description: format!(
            "The '{skill_name}' skill is a runnable automation (schedule {schedule}). \
             Accept to schedule it; the run preloads the skill."
        ),
        source: "blueprint".into(),
        spec,
        dedup_key: format!("blueprint:{skill_name}:{schedule}"),
        status: SuggestionStatus::Pending,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fill_time_blueprint_expands_minute_hour() {
        let bp = blueprints()
            .iter()
            .find(|b| b.id == "morning-brief")
            .unwrap();
        let mut values = BTreeMap::new();
        values.insert("time".to_string(), "07:30".to_string());
        let spec = bp.fill(&values);
        assert_eq!(spec.schedule, "30 7 * * *");
        assert!(spec.enabled);
        assert_eq!(spec.deliver.as_deref(), Some("origin"));
    }

    #[test]
    fn fill_weekly_blueprint_expands_weekdays() {
        let bp = blueprints()
            .iter()
            .find(|b| b.id == "weekly-review")
            .unwrap();
        let mut values = BTreeMap::new();
        values.insert("weekdays".to_string(), "Mon, Wed , Fri".to_string());
        values.insert("time".to_string(), "18:00".to_string());
        let spec = bp.fill(&values);
        assert_eq!(spec.schedule, "0 18 * * mon,wed,fri");
    }

    #[test]
    fn fill_text_slot_substitutes_prompt() {
        let bp = blueprints()
            .iter()
            .find(|b| b.id == "custom-reminder")
            .unwrap();
        let mut values = BTreeMap::new();
        values.insert("what".to_string(), "drink water".to_string());
        let spec = bp.fill(&values);
        assert_eq!(spec.payload, b"Reminder: drink water.");
    }

    #[test]
    fn fill_interval_blueprint_uses_every_grammar() {
        let bp = blueprints()
            .iter()
            .find(|b| b.id == "important-mail")
            .unwrap();
        let spec = bp.fill(&BTreeMap::new());
        assert_eq!(spec.schedule, "@every 30m");
    }

    #[test]
    fn fill_deliver_slot_local_means_store_only() {
        let bp = blueprints()
            .iter()
            .find(|b| b.id == "morning-brief")
            .unwrap();
        let mut values = BTreeMap::new();
        values.insert("deliver".to_string(), "local".to_string());
        let spec = bp.fill(&values);
        assert_eq!(spec.deliver, None);
    }

    #[test]
    fn fill_uses_defaults_when_missing() {
        let bp = blueprints()
            .iter()
            .find(|b| b.id == "morning-brief")
            .unwrap();
        let spec = bp.fill(&BTreeMap::new());
        assert_eq!(spec.schedule, "0 8 * * *");
    }

    #[test]
    fn catalog_has_fourteen_blueprints_with_unique_ids() {
        let bps = blueprints();
        assert_eq!(bps.len(), 14);
        let mut ids: Vec<_> = bps.iter().map(|b| b.id).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), 14, "blueprint ids must be unique");
    }

    #[test]
    fn starters_have_stable_unique_dedup_keys() {
        let starters = starter_suggestions();
        assert_eq!(starters.len(), 4);
        let mut keys: Vec<_> = starters.iter().map(|s| s.dedup_key.clone()).collect();
        keys.sort();
        keys.dedup();
        assert_eq!(keys.len(), 4, "dedup_keys must be unique");
        assert!(starters
            .iter()
            .all(|s| s.status == SuggestionStatus::Pending));
        assert!(starters
            .iter()
            .all(|s| s.spec.deliver.as_deref() == Some("origin")));
    }

    #[test]
    fn blueprint_skill_bridges_to_suggestion() {
        let bp = SkillBlueprint {
            schedule: "0 9 * * *".into(),
            deliver: Some("origin".into()),
            prompt: Some("Run the morning routine.".into()),
            ..SkillBlueprint::default()
        };
        let sug = blueprint_suggestion("morning-routine", &bp).unwrap();
        assert_eq!(sug.source, "blueprint");
        assert_eq!(sug.dedup_key, "blueprint:morning-routine:0 9 * * *");
        assert_eq!(sug.spec.skills, vec!["morning-routine".to_string()]);
        assert_eq!(sug.spec.deliver.as_deref(), Some("origin"));
        assert_eq!(sug.spec.payload, b"Run the morning routine.");
    }

    #[test]
    fn non_runnable_blueprint_yields_no_suggestion() {
        let bp = SkillBlueprint::default();
        assert!(blueprint_suggestion("x", &bp).is_none());
    }
}
