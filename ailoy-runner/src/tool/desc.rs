use ailoy_core::tool::MAX_TOOL_NAME_LEN;

/// Warn when `name` is longer than every model API will accept.
///
/// A warning, not an error: the target provider is unknown here.
pub(crate) fn warn_if_tool_name_too_long(name: &str) {
    if name.len() > MAX_TOOL_NAME_LEN {
        log::warn!(
            "tool name '{name}' is {} characters; some model APIs reject names over {MAX_TOOL_NAME_LEN}",
            name.len()
        );
    }
}
