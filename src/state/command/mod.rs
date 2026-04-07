mod ls;
mod mkdir;
mod rmdir;

pub use ls::run_ls;
pub use mkdir::run_mkdir;
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
