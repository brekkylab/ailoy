use crate::{
    message::{MessageDelta, Part, ToolDescription},
    tool::Tool,
};

#[derive(Clone, Debug)]
pub struct MCPTool {}

impl Tool for MCPTool {
    fn get_description(&self) -> ToolDescription {
        todo!()
    }

    fn run(
        self,
        toll_call: Part,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<MessageDelta, String>>>> {
        todo!()
    }
}
