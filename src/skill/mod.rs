//! File-based skill abstractions: portable markdown playbooks materialised
//! into a [`RunEnv`](crate::runenv::RunEnv) from per-agent specs.
//!
//! Each agent declares a list of skill directories in
//! [`AgentSpec::skills`](crate::agent::AgentSpec::skills) and, optionally, a
//! single [`AgentSpec::skill_root`](crate::agent::AgentSpec::skill_root)
//! under which it may create new skills at runtime.  Declared skills are
//! re-read on every [`Agent::snapshot`](crate::agent::Agent::snapshot); new
//! sibling skills appearing in `skill_root` are appended to the snapshot's
//! `skills` list.

mod skill;

pub use skill::*;
