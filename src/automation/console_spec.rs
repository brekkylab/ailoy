//! `console.json`: where a run's tasks execute, declared once and enforced by the
//! console server rather than by anything the tasks produce.

use cortex::{
    console::{ConsoleBuilder, NetworkAccess, SecretAccess},
    rootfs::Rootfs,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// How much of a network the session's commands get.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum NetworkReach {
    /// No network at all.
    #[default]
    None,
    /// Name resolution and the host's granted ports only.
    Host,
    /// The public internet.
    Public,
    /// Whatever the console server itself can reach.
    Full,
}

impl NetworkReach {
    pub fn as_str(self) -> &'static str {
        match self {
            NetworkReach::None => "none",
            NetworkReach::Host => "host",
            NetworkReach::Public => "public",
            NetworkReach::Full => "full",
        }
    }
}

/// The console every task of a workflow runs in.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ConsoleSpec {
    /// OCI reference of the guest image. Must carry `python3` for python tasks and
    /// `pip` when any task lists packages.
    pub image: String,

    #[serde(default)]
    pub network: NetworkReach,

    /// Credentials by reference: `{env_var, host, location}`. The value stays on the
    /// console server's side and is substituted into requests to `host` outside the
    /// guest; commands never hold it.
    #[serde(default)]
    pub secrets: Vec<SecretAccess>,
}

impl ConsoleSpec {
    /// A builder carrying the image, reach and secrets. Trees and the client are the
    /// runner's to add.
    ///
    /// `packages` are the workflow's pinned pypi packages. Baked into a rootfs recipe
    /// rather than installed at run time: the guest is torn down between tool
    /// batches, so nothing installed inside a session survives it, while a built
    /// image is cached by its recipe digest.
    pub fn console_builder(&self, packages: &[String]) -> ConsoleBuilder {
        let builder = cortex::console::Console::builder()
            .network(NetworkAccess::new(self.network.as_str()))
            .secrets(self.secrets.iter().cloned());
        match pip_install_command(packages) {
            None => builder.image(self.image.as_str()),
            Some(cmd) => builder.rootfs(Rootfs::from_image(self.image.as_str()).run(cmd)),
        }
    }

    /// A builder that leaves image, reach and secrets to the server. For a host-local
    /// console server, which can swap no image and take no network away, so that a
    /// workflow can be exercised without a micro-VM. Development only: nothing in the
    /// spec is enforced.
    pub fn host_console_builder(&self) -> ConsoleBuilder {
        cortex::console::Console::builder()
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
        assert_eq!(c.network, NetworkReach::None);
        assert!(c.secrets.is_empty());
    }

    #[test]
    fn secrets_parse_as_console_secret_access() {
        let c: ConsoleSpec = serde_json::from_str(
            r#"{"image":"i","secrets":[{"env_var":"K","host":"api.example.com","location":"query"}]}"#,
        )
        .unwrap();
        assert_eq!(c.secrets, vec![SecretAccess::query("K", "api.example.com")]);
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
    fn schema_names_the_reach_enum() {
        let schema = ConsoleSpec::schema().to_string();
        assert!(schema.contains("\"public\""), "{schema}");
    }
}
