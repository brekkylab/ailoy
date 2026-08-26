//! `headhunting` on this host directly, with no console in front of it.
//!
//! The **same `Executable`** the agent reaches. The only difference is who calls it, so
//! a command that runs here runs there.
//!
//! There are two reasons this has to exist.
//!
//! One is checking by hand. Without it, trying one change to a command means running the
//! whole app, and that calls a model every time.
//!
//! The other is the documentation check. `eval/check_transcript.py` pulls the commands
//! written in the README and runs them through here. **The example's
//! representative screen must not teach a command that does not run** — and reading it by
//! hand will not catch that.
//!
//! cortex's `cortex-execs/sqlite/src/bin/sqlite.rs` sits in the same place for the same
//! reason.

use std::{env, io::Write, process::ExitCode};

use cortex::exec::{ExecCall, Executable};
use headhunter::executable::Headhunting;

// `#[tokio::main]` is not used because the runtime it builds is more than this needs: it
// defaults to multi-threaded with an I/O driver, while what runs here is one synchronous
// query with nothing to wait on. Building it by hand below shows what is actually used.
fn main() -> ExitCode {
    // Single-threaded and with no I/O driver, because neither is needed: the query is
    // synchronous and nothing in this process waits on anything.
    let runtime = match tokio::runtime::Builder::new_current_thread().build() {
        Ok(runtime) => runtime,
        Err(e) => {
            eprintln!("headhunting: no runtime to run on: {e}");
            return ExitCode::from(1);
        }
    };
    runtime.block_on(run())
}

async fn run() -> ExitCode {
    // Where the pool is. In the app `--db` decides; here an environment variable does.
    // Either way **the command line carries no db argument** — that is the difference
    // between this command and a general-purpose `sqlite`.
    let db = env::var("HEADHUNTER_DB").unwrap_or_else(|_| "data/headhunter.db".into());

    let call = ExecCall {
        name: "headhunting".into(),
        args: env::args().skip(1).collect(),
        cwd: Some(String::new()),
        env: env::vars().collect(),
    };

    // No mount. This command opens no file in a tree — the pool is at the host path it
    // was given at registration.
    let result = Headhunting::new(db).exec(&call, None).await;

    // Written through as bytes: a caller piping this expects what was produced, on the
    // streams the program it stands in for would have used.
    std::io::stdout().write_all(&result.stdout).ok();
    std::io::stderr().write_all(&result.stderr).ok();

    ExitCode::from(result.exit_code.clamp(0, 255) as u8)
}
