use crate::message::Role;

pub trait Delta: Default {
    type Item;
    type Err;

    /// Merge `other` into `self`, like `+`.
    ///
    /// Errors when `self` and `other` are incompatible variants.
    fn accumulate(self, other: Self) -> Result<Self, Self::Err>;

    fn finish(self) -> Result<Self::Item, Self::Err>;
}

/// An error while accumulating or finishing a message or part delta.
///
/// ```
/// use ailoy::message::{Delta, DeltaError, MessageDelta, Role};
///
/// let assistant = MessageDelta::new().with_role(Role::Assistant);
/// let tool = MessageDelta::new().with_role(Role::Tool);
/// assert!(matches!(
///     assistant.accumulate(tool),
///     Err(DeltaError::RoleMismatch { left: Role::Assistant, right: Role::Tool })
/// ));
/// ```
#[derive(Debug, thiserror::Error)]
pub enum DeltaError {
    #[error("Cannot accumulate two message deltas with different roles. ({left} != {right})")]
    RoleMismatch { left: Role, right: Role },
    #[error("Role not specified")]
    MissingRole,
    #[error("Cannot accumulate two functions with different ids. ({left} != {right}).")]
    FunctionIdMismatch { left: String, right: String },

    /// The part types or the function fragment encodings cannot be combined.
    #[error("{details}")]
    IncompatibleParts { details: String },

    /// A `Verbatim` call also requires a string `name` and an `arguments` field.
    #[error("tool-call arguments are not valid JSON")]
    InvalidToolCallJson,
}
