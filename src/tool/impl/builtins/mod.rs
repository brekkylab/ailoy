mod apply_patch;
mod edit;
mod imgread;
mod read;
mod read_file;
mod shell;
mod web_fetch;
mod web_search;
mod write;

use std::sync::Arc;

pub use apply_patch::*;
pub use edit::*;
pub use imgread::*;
pub use read::*;
pub use read_file::*;
pub use shell::*;
pub use web_fetch::*;
pub use web_search::*;
pub use write::*;

use crate::tool::ToolFactory;

/// An `n` in `0..=max` for which `fits(n)` holds and, short of `max`, `fits(n + 1)` does not;
/// the largest such `n` when `fits` holds for every `n` below one it holds for. `None` when
/// neither `fits(max)` nor `fits(0)` holds.
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

/// Build `(name, factory)` pairs for every built-in tool. The factory takes the
/// [`ToolDesc`](crate::tool::ToolDesc) requested by an agent spec and returns the
/// [`ToolFunc`](crate::tool::ToolFunc) to bind to it.
pub fn get_builtin_tool_factories() -> Vec<(&'static str, ToolFactory)> {
    let shell = get_shell_tool_func();
    let read = get_read_tool_func();
    let read_file = get_read_file_tool_func();
    let imgread = get_imgread_tool_func();
    let write = get_write_tool_func();
    let edit = get_edit_tool_func();
    let apply_patch = get_apply_patch_tool_func();

    vec![
        ("shell", Arc::new(move |_| Ok(shell.clone()))),
        ("read", Arc::new(move |_| Ok(read.clone()))),
        ("read_file", Arc::new(move |_| Ok(read_file.clone()))),
        ("imgread", Arc::new(move |_| Ok(imgread.clone()))),
        ("write", Arc::new(move |_| Ok(write.clone()))),
        ("edit", Arc::new(move |_| Ok(edit.clone()))),
        ("apply_patch", Arc::new(move |_| Ok(apply_patch.clone()))),
        ("web_search", Arc::new(get_web_search_tool_factory(vec![]))),
        ("web_fetch", Arc::new(get_web_fetch_tool_factory())),
    ]
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
            }
        }
    }

    /// When `fits` fails below `max` and holds again above the gap, the `n` found still fits and
    /// `n + 1` does not; with neither end fitting, nothing is found.
    #[test]
    fn largest_fitting_stops_where_the_next_does_not_fit() {
        for max in 0..=12 {
            for gap in 0..=12 {
                let fits = |n: usize| n < 3 || (gap < n && n < max);
                let n = largest_fitting(max, fits).unwrap();
                assert!(fits(n), "max {max}, gap {gap}: {n}");
                assert!(n == max || !fits(n + 1), "max {max}, gap {gap}: {n}");
            }
        }
        assert_eq!(largest_fitting(5, |n| n == 3), None);
        assert_eq!(largest_fitting(5, |n| n != 0), Some(5));
    }
}
