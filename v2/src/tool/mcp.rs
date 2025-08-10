use crate::{
    tool::Tool,
    value::{Part, ToolDescription},
};

#[derive(Clone, Debug)]
pub struct MCPTool {}

impl Tool for MCPTool {
    fn get_description(&self) -> ToolDescription {
        todo!()
    }

    fn run(self, _: Part) -> std::pin::Pin<Box<dyn Future<Output = Result<Part, String>>>> {
        todo!()
    }
}
