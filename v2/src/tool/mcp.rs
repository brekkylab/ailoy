use crate::{
    tool::Tool,
    value::{Part, ToolCall, ToolDescription},
};

#[derive(Clone, Debug)]
pub struct MCPTool {}

impl Tool for MCPTool {
    fn get_description(&self) -> ToolDescription {
        todo!()
    }

    fn run(self, _: ToolCall) -> std::pin::Pin<Box<dyn Future<Output = Result<Part, String>>>> {
        todo!()
    }
}
