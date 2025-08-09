use crate::{
    message::{MessageDelta, Part},
    tool::Tool,
};

#[derive(Clone, Debug)]
pub struct MCPTool {}

impl Tool for MCPTool {
    fn get_description(&self) -> Part {
        todo!()
    }

    fn run(
        self,
        toll_call: Part,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<MessageDelta, String>>>> {
        todo!()
    }
}
