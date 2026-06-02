//! File-based skill abstractions: portable markdown playbooks materialised
//! into a [`RunEnv`](crate::runenv::RunEnv) from per-agent specs.
//!
//! Each agent declares a list of skill directories in
//! [`AgentSpec::skills`](crate::agent::AgentSpec::skills) with their
//! pre-fill content carried in [`AgentSpec::files`](crate::agent::AgentSpec::files).
//! At agent construction those files are written into the runenv with
//! write-once semantics; the system instruction lists the declared skills via
//! [`AgentState::render_skills`](crate::agent::AgentState::render_skills),
//! which loads each skill's metadata with [`get_skill`].

#[allow(clippy::module_inception)]
mod skill;

pub use skill::*;
