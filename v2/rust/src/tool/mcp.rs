use crate::tool::Tool;

#[derive(Clone, Debug)]
pub struct MCPTool {}

impl Tool for MCPTool {
    fn get_description(&self) -> crate::Part {
        todo!()
    }

    fn run(
        self,
        toll_call: crate::Part,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<crate::MessageDelta, String>>>> {
        todo!()
    }
}
