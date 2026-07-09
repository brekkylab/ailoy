//! File-based skill abstractions: portable markdown playbooks materialised
//! into a [`Machine`](crate::runenv::Machine) from per-agent specs.
//!
//! Each agent declares a list of skill directories in
//! [`AgentSpec::skills`](crate::agent::AgentSpec::skills) with their
//! pre-fill content carried in [`AgentSpec::files`](crate::agent::AgentSpec::files).
//! On the first [`Agent::run`](crate::agent::Agent::run) those files are
//! written into the machine with write-once semantics; the system instruction
//! lists the declared skills via [`scan_declared_skills`].

#[allow(clippy::module_inception)]
mod skill;

pub use skill::*;
