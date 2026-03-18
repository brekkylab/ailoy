// Allow proc-macros to use `ailoy::` prefix when invoked from within this crate.
extern crate self as ailoy;

pub mod agent;
pub mod datatype;
pub mod message;

pub use agent::{
    AgentProvider, AgentRuntime, AgentSpec, LangModelAPISchema, LangModelProvider, ToolProvider,
    ToolRuntime, ToolSet,
};
pub use datatype::Value;
pub use message::{
    Message, MessageDeltaOutput, MessageOutput, Part, Role, ToolDesc, ToolDescBuilder,
};
