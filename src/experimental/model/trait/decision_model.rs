use futures::future::BoxFuture;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::datatype::Value;

pub trait DecisionModelInference: Send + Sync {
    /// Answers each of `questions` about `state`, keyed by the same ids.
    fn infer<'a>(
        &'a self,
        state: &'a Value,
        questions: &'a IndexMap<String, Question>,
    ) -> BoxFuture<'a, anyhow::Result<IndexMap<String, Answer>>>;
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Question {
    /// Picks one of the named options; each may carry a description.
    Choice {
        instructions: String,
        criteria: IndexMap<String, Option<String>>,
    },
    /// Rates on levels `0..criteria.len()`, each level described by its criterion.
    Score {
        instructions: String,
        criteria: Vec<String>,
    },
    /// Yes/no: the probability that the statement in `instructions` holds.
    Noul {
        instructions: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        criteria: Option<NoulCriteria>,
    },
}

/// What true and false mean for a [`Question::Noul`]; either may be left to the model's default.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct NoulCriteria {
    #[serde(rename = "true", skip_serializing_if = "Option::is_none")]
    pub when_true: Option<String>,
    #[serde(rename = "false", skip_serializing_if = "Option::is_none")]
    pub when_false: Option<String>,
}

impl Question {
    pub fn choice(
        instructions: impl Into<String>,
        criteria: IndexMap<String, Option<String>>,
    ) -> anyhow::Result<Self> {
        let instructions = non_empty(instructions)?;
        anyhow::ensure!(criteria.len() >= 2, "a choice needs two or more criteria");
        Ok(Self::Choice {
            instructions,
            criteria,
        })
    }

    pub fn score(instructions: impl Into<String>, criteria: Vec<String>) -> anyhow::Result<Self> {
        let instructions = non_empty(instructions)?;
        anyhow::ensure!(criteria.len() >= 2, "a score needs two or more levels");
        Ok(Self::Score {
            instructions,
            criteria,
        })
    }

    pub fn noul(
        instructions: impl Into<String>,
        criteria: Option<NoulCriteria>,
    ) -> anyhow::Result<Self> {
        Ok(Self::Noul {
            instructions: non_empty(instructions)?,
            criteria,
        })
    }

    pub fn instructions(&self) -> &str {
        match self {
            Self::Choice { instructions, .. }
            | Self::Score { instructions, .. }
            | Self::Noul { instructions, .. } => instructions,
        }
    }
}

fn non_empty(instructions: impl Into<String>) -> anyhow::Result<String> {
    let instructions = instructions.into();
    anyhow::ensure!(!instructions.trim().is_empty(), "instructions are missing");
    Ok(instructions)
}

/// The model's answer to one [`Question`], of the same type.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Answer {
    #[serde(flatten)]
    pub value: AnswerValue,
    /// 1 minus the normalized entropy of the answer's distribution: 0 is a coin toss, 1 is certain.
    pub confidence: f64,
    pub action: Action,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnswerValue {
    /// The most probable option, and the probability of every option.
    Choice {
        choice: String,
        probabilities: IndexMap<String, f64>,
    },
    /// The expected level, and the probability of each level by index.
    Score { score: f64, probabilities: Vec<f64> },
    /// The probability that the statement holds.
    Noul { noul: f64 },
}

/// Whether to act on the answer rather than escalate it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Action {
    pub act_probability: f64,
}
