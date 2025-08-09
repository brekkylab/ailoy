use std::{fmt::Debug, sync::Arc};

use crate::{
    tool::Tool,
    value::{MessageDelta, Part, ToolDescription},
};

#[derive(Clone)]
pub struct BuiltinTool {
    desc: ToolDescription,
    behavior: Arc<dyn Fn() -> ()>,
}

impl Debug for BuiltinTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BuiltinTool")
            .field("desc", &self.desc)
            .field("behavior", &())
            .finish()
    }
}

impl BuiltinTool {}

impl Tool for BuiltinTool {
    fn get_description(&self) -> ToolDescription {
        self.desc.clone()
    }

    fn run(
        self,
        toll_call: Part,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<MessageDelta, String>>>> {
        let tool_call = toll_call.get_json();
        todo!()
    }
}
