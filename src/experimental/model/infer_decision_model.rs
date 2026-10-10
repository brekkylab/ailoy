use futures::future::BoxFuture;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::datatype::Value;

/// A typed question put to an [`InferDecisionModel`] about a state, in the shape Jev and Laya take.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum DecisionQuestion {
    /// Pick one of the labels, with a probability for each.
    Choice {
        instructions: Value,
        /// Each label and what it covers; `Value::Null` leaves the label to speak for itself.
        criteria: IndexMap<String, Value>,
    },
    /// Pick a level on an ordered scale, answered as its expected value.
    Score {
        instructions: Value,
        /// What each level means, from lowest to highest.
        criteria: Vec<Value>,
    },
    /// A yes or no question, answered as the probability of yes.
    Noul {
        instructions: Value,
        /// What `true` and `false` mean; `None` leaves them to the instructions.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        criteria: Option<NoulCriteria>,
    },
}

/// What a yes and a no mean for a [`DecisionQuestion::Noul`].
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct NoulCriteria {
    #[serde(rename = "true", default, skip_serializing_if = "Option::is_none")]
    pub yes: Option<Value>,
    #[serde(rename = "false", default, skip_serializing_if = "Option::is_none")]
    pub no: Option<Value>,
}

/// Whether to act on an answer rather than escalate it, from models that say so (Laya).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DecisionAction {
    pub act_probability: f64,
}

/// The model's answer to one [`DecisionQuestion`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum DecisionAnswer {
    Choice {
        /// The label picked.
        choice: String,
        /// The probability of every label.
        probabilities: IndexMap<String, f64>,
        /// How peaked the distribution is, between 0 and 1.
        confidence: f64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        action: Option<DecisionAction>,
    },
    Score {
        /// The expected level, from 0 for the lowest.
        score: f64,
        /// The probability of every level, keyed by its index (`"0"`, `"1"`, …).
        probabilities: IndexMap<String, f64>,
        /// What each level means, keyed by its index.
        #[serde(default)]
        legend: IndexMap<String, String>,
        confidence: f64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        action: Option<DecisionAction>,
    },
    Noul {
        /// The probability of yes.
        noul: f64,
        /// Given by Laya, not by Jev.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        confidence: Option<f64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        action: Option<DecisionAction>,
    },
}

/// A model that answers typed questions about a state, and generates no text.
pub trait InferDecisionModel: Send + Sync {
    /// Answer every question about `state`, keyed by the ids they were given.
    ///
    /// A question the model cannot take comes back as an error under its id, and the others are
    /// still answered.
    fn infer(
        &self,
        state: &Value,
        questions: &IndexMap<String, DecisionQuestion>,
    ) -> BoxFuture<'static, anyhow::Result<IndexMap<String, anyhow::Result<DecisionAnswer>>>>;
}
