//! File-based skill abstractions: portable markdown playbooks materialised
//! into a [`RunEnv`] from per-agent specs.
//!
//! See [`Skill`] for the data structure that lives on
//! [`AgentSpec`](crate::agent::AgentSpec). Each agent owns a directory inside
//! the runenv at which its declared skills are written and from which runtime
//! changes can be read back via [`Agent::snapshot`](crate::agent::Agent::snapshot).

mod skill;

pub use skill::*;
