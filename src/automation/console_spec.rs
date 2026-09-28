//! `console.json`: where a run's tasks execute, declared once and enforced by the
//! console server rather than by anything the tasks produce.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use virtx::{console::ConsoleClientBuilder, image::Recipe};

/// Where a run's scratch tree is, as a command in the console spells it.
pub const GUEST_SCRATCH: &str = "/scratch";
/// Where a run's artifacts tree is, as a command in the console spells it.
pub const GUEST_ARTIFACTS: &str = "/artifacts";

/// The console every task of a workflow runs in.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ConsoleSpec {
    /// OCI reference of the guest image. Must carry `python3` for python tasks and
    /// `pip` when any task lists packages.
    pub image: String,

    /// Whether the session's commands reach a network at all.
    #[serde(default)]
    pub network: bool,
}

impl ConsoleSpec {
    /// A builder carrying the image and the network answer. Mounts and the client are
    /// the runner's to add.
    ///
    /// `packages` are the workflow's pinned pypi packages. Baked into the image recipe
    /// rather than installed at run time: the guest is torn down between tool batches,
    /// so nothing installed inside a session survives it, while a built image is cached
    /// by its recipe digest.
    pub fn console_builder(&self, packages: &[String]) -> ConsoleClientBuilder {
        let builder = virtx::console::ConsoleClient::builder().network(self.network);
        match pip_install_command(packages) {
            None => builder.image(virtx::image::ImageSource::reference(self.image.as_str())),
            Some(cmd) => builder.image(Recipe::new(self.image.as_str()).step(cmd)),
        }
    }

    pub fn schema() -> serde_json::Value {
        serde_json::to_value(schemars::schema_for!(ConsoleSpec)).expect("schema serializes")
    }
}

/// Sorted and deduplicated, so the same package set is the same recipe digest.
pub(super) fn pip_install_command(packages: &[String]) -> Option<String> {
    let mut pkgs: Vec<&str> = packages.iter().map(String::as_str).collect();
    pkgs.sort_unstable();
    pkgs.dedup();
    if pkgs.is_empty() {
        return None;
    }
    Some(format!("pip install --no-cache-dir {}", pkgs.join(" ")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimal_spec_fills_defaults() {
        let c: ConsoleSpec = serde_json::from_str(r#"{"image":"python:3.12-slim"}"#).unwrap();
        assert!(!c.network);
    }

    #[test]
    fn pip_command_is_sorted_and_deduplicated() {
        let pkgs = ["b==2".to_string(), "a==1".to_string(), "b==2".to_string()];
        assert_eq!(
            pip_install_command(&pkgs).unwrap(),
            "pip install --no-cache-dir a==1 b==2"
        );
        assert!(pip_install_command(&[]).is_none());
    }

    #[test]
    fn schema_names_the_fields() {
        let schema = ConsoleSpec::schema().to_string();
        assert!(
            schema.contains("\"image\"") && schema.contains("\"network\""),
            "{schema}"
        );
    }
}
