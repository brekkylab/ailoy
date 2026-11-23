//! Where an agent runs. An [`AgentProvider`] holds what an [`AgentSpec`] only names, the
//! providers of its tools, and the agents kept by name, each with its sessions. It is served
//! over HTTP by two routers: [`router`], to `ailoy_core::Client`, and [`admin_router`], to
//! `ailoy_core::AdminClient`.
//!
//! [`AgentSpec`]: ailoy_core::agent::AgentSpec

mod provider;
mod router;

pub use provider::{AgentProvider, Refusal};
pub use router::{admin_router, router};
