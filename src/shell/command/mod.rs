mod cat;
mod cp;
mod ls;
mod mkdir;
mod mv;
mod pwd;
mod rm;
mod rmdir;

pub use cat::run_cat;
pub use cp::run_cp;
pub use ls::run_ls;
pub use mkdir::run_mkdir;
pub use mv::run_mv;
pub use pwd::run_pwd;
pub use rm::run_rm;
pub use rmdir::run_rmdir;

pub struct ExecOutput {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
}

impl ExecOutput {
    pub fn ok(stdout: impl Into<String>) -> Self {
        Self { stdout: stdout.into(), stderr: String::new(), exit_code: 0 }
    }

    pub fn err(exit_code: i32, stderr: impl Into<String>) -> Self {
        Self { stdout: String::new(), stderr: stderr.into(), exit_code }
    }
}
