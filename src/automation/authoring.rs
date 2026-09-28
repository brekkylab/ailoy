//! The reference material an agent writes an automation from: the guide and the
//! schemas its files are checked against.
//!
//! Carried by the crate rather than read from a repository, so a caller that puts an
//! agent in a console can write these into whatever the agent can reach.

use serde_json::Value;

use crate::automation::{ConsoleSpec, Workflow};

/// How to write an automation: the directory's files, the task contracts, the rules
/// `AutomationDef::load` enforces and the message each one fails with.
pub const SKILL: &str = include_str!("SKILL.md");

pub const SKILL_FILE: &str = "SKILL.md";
pub const CONSOLE_SCHEMA_FILE: &str = "console.schema.json";
pub const WORKFLOW_SCHEMA_FILE: &str = "workflow.schema.json";

/// The guide and the schemas as `(file name, contents)`, ready to be written where an
/// agent can read them.
///
/// `trigger.json` is the daemon's registration body and its schema comes from the
/// daemon (`ailoy-daemon schema`), not from here.
pub fn authoring_files() -> Vec<(&'static str, String)> {
    vec![
        (SKILL_FILE, SKILL.to_string()),
        (CONSOLE_SCHEMA_FILE, pretty(ConsoleSpec::schema())),
        (WORKFLOW_SCHEMA_FILE, pretty(Workflow::schema())),
    ]
}

fn pretty(v: Value) -> String {
    serde_json::to_string_pretty(&v).expect("a schema serializes")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_guide_and_schemas_come_from_the_crate() {
        let files = authoring_files();
        assert_eq!(files.len(), 3);
        assert!(SKILL.contains("# Writing an ailoy automation"), "{SKILL}");
        for (name, body) in files {
            assert!(!body.is_empty(), "{name} is empty");
            if name.ends_with(".json") {
                let v: Value = serde_json::from_str(&body).expect(name);
                assert!(
                    v.get("$defs").is_some() || v.get("properties").is_some(),
                    "{name}"
                );
            }
        }
    }
}
