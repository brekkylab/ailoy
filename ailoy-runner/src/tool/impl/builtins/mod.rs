mod apply_patch;
mod browser;
mod edit;
mod imgread;
mod read;
mod read_file;
mod shell;
mod web_fetch;
mod web_search;
mod write;

pub use apply_patch::*;
pub use browser::*;
pub use edit::*;
pub use imgread::*;
pub use read::*;
pub use read_file::*;
pub use shell::*;
pub use web_fetch::*;
pub use web_search::*;
pub use write::*;

use crate::tool::{ToolDesc, ToolFunc, ToolProvider};

/// Binary-searches `0..=max` for the largest `n` with `fits(n)`, assuming `fits` stays
/// true below any `n < max` it is true for. `None` when nothing fits, not even `n == 0`.
fn largest_fitting(max: usize, fits: impl Fn(usize) -> bool) -> Option<usize> {
    if fits(max) {
        return Some(max);
    }
    if !fits(0) {
        return None;
    }
    // `fits(lo)` holds and `fits(hi)` does not.
    let (mut lo, mut hi) = (0, max);
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if fits(mid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Some(lo)
}

/// Every built-in tool, as one provider.
pub struct Builtins {
    /// Each tool's description, and what makes it each time an agent is built with it.
    #[allow(clippy::type_complexity)]
    tools: Vec<(
        ToolDesc,
        Box<dyn Fn() -> anyhow::Result<ToolFunc> + Send + Sync>,
    )>,
}

impl Builtins {
    pub fn new() -> Self {
        /// A tool every agent shares.
        fn shared(func: ToolFunc) -> Box<dyn Fn() -> anyhow::Result<ToolFunc> + Send + Sync> {
            Box::new(move || Ok(func.clone()))
        }
        Self {
            tools: vec![
                (get_shell_tool_desc(), shared(get_shell_tool_func())),
                (get_read_tool_desc(), shared(get_read_tool_func())),
                (get_read_file_tool_desc(), shared(get_read_file_tool_func())),
                (get_imgread_tool_desc(), shared(get_imgread_tool_func())),
                (get_write_tool_desc(), shared(get_write_tool_func())),
                (get_edit_tool_desc(), shared(get_edit_tool_func())),
                (
                    get_apply_patch_tool_desc(),
                    shared(get_apply_patch_tool_func()),
                ),
                (get_browser_tool_desc(), shared(get_browser_tool_func())),
                (
                    get_browser_screenshot_tool_desc(),
                    shared(get_browser_screenshot_tool_func()),
                ),
                // Made for each agent: it searches with the provider set at the time.
                (
                    get_web_search_tool_desc(),
                    Box::new(get_web_search_tool_factory()),
                ),
                (
                    get_web_fetch_tool_desc(),
                    Box::new(get_web_fetch_tool_factory()),
                ),
            ],
        }
    }
}

impl ToolProvider for Builtins {
    fn descs(&self) -> Vec<ToolDesc> {
        self.tools.iter().map(|(desc, _)| desc.clone()).collect()
    }

    fn make(&self, name: &str) -> anyhow::Result<ToolFunc> {
        let (_, factory) = self
            .tools
            .iter()
            .find(|(desc, _)| desc.name == name)
            .ok_or_else(|| anyhow::anyhow!("'{name}' is no built-in tool"))?;
        factory()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn largest_fitting_finds_the_last_n_that_fits() {
        for max in 0..=12 {
            // `fits(n)` holds for `n < fitting`.
            for fitting in 0..=14 {
                assert_eq!(
                    largest_fitting(max, |n| n < fitting),
                    fitting.checked_sub(1).map(|last| last.min(max)),
                    "max {max}, {fitting} fitting"
                );
                // `max` fits too, whether or not the `n` below it does.
                assert_eq!(
                    largest_fitting(max, |n| n < fitting || n == max),
                    Some(max),
                    "max {max}, {fitting} fitting and max"
                );
            }
        }
    }
}
