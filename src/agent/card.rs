use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Public-facing descriptor that an agent exposes to its callers.
///
/// An `AgentCard` is *outward-facing* metadata — it tells a calling agent (or
/// orchestrator) what this agent is called, what it does, and which skills it
/// offers.  This is intentionally distinct from a system instruction: a system
/// instruction is private, internal guidance that shapes how the agent reasons
/// and behaves; an `AgentCard` is consumed by the *caller* to decide whether
/// and how to delegate work to this agent.
///
/// The structure is inspired by and compatible with the A2A agent-card format.
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct AgentCard {
    /// Human-readable name that identifies this agent.
    pub name: String,

    /// Short description of what the agent does, written for a caller that
    /// needs to decide whether to route a task here.
    pub description: String,

    /// Discrete capabilities this agent can perform.  An empty list means the
    /// agent exposes no structured skills; the description alone describes it.
    #[serde(default)]
    pub skills: Vec<AgentSkill>,
}

/// A single, named capability advertised by an [`AgentCard`].
///
/// Skills let callers match a specific subtask to the right agent without
/// parsing free-form descriptions.  Each skill should map to a concrete,
/// well-scoped action the agent can execute.
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct AgentSkill {
    /// Stable machine-readable identifier for this skill (e.g. `"web_search"`).
    /// Should not change once the agent is published.
    pub id: String,

    /// Human-readable display name for the skill (e.g. `"Web Search"`).
    pub name: String,

    /// Description of what the skill does, written for the calling agent.
    pub description: String,
}
