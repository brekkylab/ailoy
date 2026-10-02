use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Outward-facing descriptor a caller reads to decide whether and how to delegate to this
/// agent; unlike the system instruction, it is not guidance to the agent itself.
///
/// Compatible with the A2A agent-card format.
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct AgentCard {
    /// Human-readable name that identifies this agent.
    pub name: String,

    /// Short description of what the agent does, written for a caller that
    /// needs to decide whether to route a task here.
    pub description: String,

    /// Empty when the description alone describes the agent.
    #[serde(default)]
    pub skills: Vec<AgentSkill>,
}

/// A named capability of an [`AgentCard`], so a caller can match a subtask without parsing
/// the description.
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
