use super::*;

const ARXIV: &str = "---\nname: arxiv\ndescription: \"Search arXiv papers by keyword, author, category, or ID.\"\nversion: 1.0.0\nplatforms: [linux, macos, windows]\nmetadata:\n  hermes:\n    tags: [Research, Arxiv, Papers]\n    related_skills: [ocr-and-documents]\n---\n\n# arXiv\n\n## When to Use\nWhen searching academic papers.\n";

fn store() -> (tempfile::TempDir, SkillStore) {
    let dir = tempfile::tempdir().unwrap();
    let store = SkillStore::new(dir.path().join("skills"));
    (dir, store)
}

#[test]
fn parses_frontmatter_fields() {
    let fm = parse_frontmatter(ARXIV).expect("parse");
    assert_eq!(fm.name, "arxiv");
    assert!(fm.description.starts_with("Search arXiv"));
    assert_eq!(fm.version.as_deref(), Some("1.0.0"));
    assert_eq!(fm.platforms, vec!["linux", "macos", "windows"]);
    assert_eq!(fm.hermes().tags, vec!["Research", "Arxiv", "Papers"]);
    assert_eq!(fm.hermes().related_skills, vec!["ocr-and-documents"]);
}

#[test]
fn create_discover_and_index() {
    let (_d, store) = store();
    store.create("arxiv", ARXIV, Some("research")).unwrap();

    let entries = store.discover();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].name, "arxiv");
    assert_eq!(entries[0].category.as_deref(), Some("research"));

    let index = store.render_index();
    assert!(index.contains("## Skills (mandatory)"));
    assert!(index.contains("<available_skills>"));
    assert!(index.contains("  research:"));
    assert!(index.contains("    - arxiv: Search arXiv papers"));
}

#[test]
fn index_is_empty_without_skills() {
    let (_d, store) = store();
    assert_eq!(store.render_index(), "");
}

#[test]
fn create_rejects_duplicate_and_malformed() {
    let (_d, store) = store();
    store.create("arxiv", ARXIV, Some("research")).unwrap();
    assert!(matches!(
        store.create("arxiv", ARXIV, Some("research")),
        Err(SkillError::Exists(_))
    ));
    assert!(matches!(
        store.create("bad", "no frontmatter here", None),
        Err(SkillError::Malformed(_))
    ));
}

#[test]
fn view_full_body_and_linked_file() {
    let (_d, store) = store();
    store.create("arxiv", ARXIV, Some("research")).unwrap();
    let body = store.view("arxiv", None).unwrap();
    assert!(body.contains("## When to Use"));

    store
        .write_file("arxiv", "references/api.md", "# API\nendpoints")
        .unwrap();
    let linked = store.view("arxiv", Some("references/api.md")).unwrap();
    assert!(linked.contains("endpoints"));
}

#[test]
fn patch_edits_in_place_and_misses_are_errors() {
    let (_d, store) = store();
    store.create("arxiv", ARXIV, Some("research")).unwrap();
    store
        .patch("arxiv", "academic papers", "scholarly works", None, false)
        .unwrap();
    assert!(store.view("arxiv", None).unwrap().contains("scholarly works"));
    assert!(matches!(
        store.patch("arxiv", "does-not-exist", "x", None, false),
        Err(SkillError::PatchMiss(_))
    ));
}

#[test]
fn write_file_path_guard() {
    let (_d, store) = store();
    store.create("arxiv", ARXIV, Some("research")).unwrap();
    // Traversal + non-support dirs are rejected.
    assert!(matches!(
        store.write_file("arxiv", "../escape.txt", "x"),
        Err(SkillError::Invalid(_))
    ));
    assert!(matches!(
        store.write_file("arxiv", "secrets/key", "x"),
        Err(SkillError::Invalid(_))
    ));
}

#[test]
fn delete_removes_bundle() {
    let (_d, store) = store();
    store.create("arxiv", ARXIV, Some("research")).unwrap();
    store.delete("arxiv").unwrap();
    assert!(store.discover().is_empty());
    assert!(matches!(
        store.delete("arxiv"),
        Err(SkillError::NotFound(_))
    ));
}

#[test]
fn cache_invalidated_on_write() {
    let (_d, store) = store();
    store.create("arxiv", ARXIV, Some("research")).unwrap();
    let first = store.render_index();
    assert!(first.contains("arxiv"));
    store
        .create(
            "obsidian",
            &ARXIV.replace("arxiv", "obsidian").replace("Search arXiv", "Manage Obsidian"),
            Some("productivity"),
        )
        .unwrap();
    let second = store.render_index();
    assert!(second.contains("obsidian"), "index re-rendered after write");
}

#[test]
fn seed_from_bundled_skips_existing() {
    let bundled_tmp = tempfile::tempdir().unwrap();
    let bundled = SkillStore::new(bundled_tmp.path().join("skills"));
    bundled.create("arxiv", ARXIV, Some("research")).unwrap();
    bundled
        .create(
            "maps",
            &ARXIV.replace("arxiv", "maps").replace("Search arXiv", "Driving directions"),
            Some("productivity"),
        )
        .unwrap();

    let (_d, user) = store();
    // User already has a (customized) arxiv; seeding must not clobber it but should add maps.
    user.create(
        "arxiv",
        &ARXIV.replace("Search arXiv", "MY custom arxiv"),
        Some("research"),
    )
    .unwrap();

    let seeded = user.seed_from(bundled.root()).unwrap();
    assert_eq!(seeded, vec!["maps".to_string()]);
    assert!(user.view("arxiv", None).unwrap().contains("MY custom arxiv"));
    assert_eq!(user.discover().len(), 2);
}

#[test]
fn seed_bundled_materializes_curated_skills() {
    let (_d, store) = store();
    let seeded = store.seed_bundled().unwrap();

    // The curated set ships these portable, tool-agnostic skills.
    for expected in ["plan", "systematic-debugging", "design-md", "research-paper-writing"] {
        assert!(seeded.contains(&expected.to_string()), "missing {expected}");
    }
    // Categories are derived from the embedded path layout.
    let entries = store.discover();
    let plan = entries.iter().find(|e| e.name == "plan").unwrap();
    assert_eq!(plan.category.as_deref(), Some("software-development"));
    // Linked reference files come along (progressive disclosure level 3).
    let refs = store
        .view("research-paper-writing", Some("references/writing-guide.md"))
        .unwrap();
    assert!(!refs.is_empty());

    // Idempotent + non-clobbering: a re-seed adds nothing.
    let again = store.seed_bundled().unwrap();
    assert!(again.is_empty(), "re-seed should be a no-op, got {again:?}");
}
