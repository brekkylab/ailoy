// Allow proc-macros to use `ailoy::` prefix when invoked from within this crate.
extern crate self as ailoy;

pub mod agent;
pub mod datatype;
pub mod message;
pub mod shell;

pub use agent::{
    AgentProvider, AgentRuntime, AgentSpec, BuiltinToolProvider, LangModelAPISchema,
    LangModelInferConfig, LangModelProvider, MCPToolProvider, ToolAsyncFunc, ToolProvider,
    ToolRuntime, ToolSet, ToolStreamingFunc, ToolSyncFunc, TurnEvent,
};
pub use datatype::Value;
pub use message::{
    Message, MessageDeltaOutput, MessageOutput, Part, Role, StreamingToolOutput, ToolDesc,
    ToolDescBuilder, ToolResultDelta,
};
pub use shell::Shell;
