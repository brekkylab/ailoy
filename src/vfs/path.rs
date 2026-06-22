/// A mount-relative virtual path, always absolute within its mount
/// (leading `/`, no trailing `/` except for the root `/`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VPath(String);

impl VPath {
    pub fn new(s: impl AsRef<str>) -> Self {
        let s = s.as_ref();
        let trimmed = s.trim_end_matches('/');
        if trimmed.is_empty() {
            VPath("/".to_string())
        } else if let Some(stripped) = trimmed.strip_prefix('/') {
            VPath(format!("/{stripped}"))
        } else {
            VPath(format!("/{trimmed}"))
        }
    }

    pub fn root() -> Self {
        VPath("/".to_string())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn is_root(&self) -> bool {
        self.0 == "/"
    }
}
