use std::{
    fmt::{self, Debug, Formatter},
    pin::Pin,
    sync::Arc,
};

use crate::{
    tool::Tool,
    value::{Part, ToolDescription, ToolDescriptionArgument},
};

#[derive(Clone)]
pub struct BuiltinTool {
    desc: ToolDescription,
    f: Arc<dyn Fn(&serde_json::Value) -> Part>,
}

impl Debug for BuiltinTool {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("BuiltinTool")
            .field("desc", &self.desc)
            .field("f", &"(Function)")
            .finish()
    }
}

impl BuiltinTool {}

impl Tool for BuiltinTool {
    fn get_description(&self) -> ToolDescription {
        self.desc.clone()
    }

    fn run(self, toll_call: Part) -> Pin<Box<dyn Future<Output = Result<Part, String>>>> {
        Box::pin(async move { Ok((self.f)(toll_call.get_json().unwrap())) })
    }
}

pub fn create_terminal_tool() {
    let current_shell = {
        #[cfg(target_family = "unix")]
        {
            "bash"
        }
        #[cfg(target_family = "windows")]
        {
            "powershell"
        }
    };
    let desc = ToolDescription::new(
        "execute_command",
        format!(
            "Executes a command on current system. The current shell is {}.",
            current_shell
        ),
        ToolDescriptionArgument::new_object().with_properties(
            [(
                "command",
                ToolDescriptionArgument::new_string().with_desc("The command to execute."),
            )],
            ["command"],
        ),
        ToolDescriptionArgument::new_string().with_desc("stdout results"),
    );
}
