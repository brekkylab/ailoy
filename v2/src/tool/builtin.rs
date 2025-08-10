use std::{
    fmt::{self, Debug, Formatter},
    pin::Pin,
    sync::Arc,
};

use crate::{
    tool::Tool,
    value::{Part, ToolCall, ToolDescription, ToolDescriptionArgument},
};

#[derive(Clone)]
pub struct BuiltinTool {
    desc: ToolDescription,
    f: Arc<dyn Fn(ToolCall) -> Part>,
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

    fn run(self, toll_call: ToolCall) -> Pin<Box<dyn Future<Output = Result<Part, String>>>> {
        Box::pin(async move { Ok((self.f)(toll_call)) })
    }
}

pub fn create_terminal_tool() -> BuiltinTool {
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
        Some(ToolDescriptionArgument::new_string().with_desc("stdout results")),
    );
    let f = Arc::new(|tc: ToolCall| {
        let args = tc.get_argument().as_object().unwrap();
        let cmd = args.get("command").unwrap().as_str().unwrap();
        Part::new_text(cmd.to_owned())
    });
    BuiltinTool { desc, f }
}
