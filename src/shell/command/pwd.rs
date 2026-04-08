use crate::shell::Shell;

use super::ExecOutput;

pub fn run_pwd(shell: &Shell, _args: &[&str]) -> ExecOutput {
    if shell.cwd.is_empty() {
        ExecOutput::ok("/")
    } else {
        ExecOutput::ok(format!("/{}", shell.cwd.join("/")))
    }
}
