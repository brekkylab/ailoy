//! The tools this crate provides, each group a [`ToolProvider`](crate::tool::ToolProvider):
//! [`Console`] in the agent's sandbox, [`Web`] from this process, [`Browser`] over the sandbox's
//! `browser` abin, and an MCP server's ([`MCPServer`]).

mod browser;
mod console;
mod mcp;
mod web;

pub use browser::*;
pub use console::*;
pub(crate) use mcp::prefixed_tool_name;
pub use mcp::{MCP_NAME_SEPARATOR, MCPConnection, MCPServer, MCPToolEntry};
pub use web::*;

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
