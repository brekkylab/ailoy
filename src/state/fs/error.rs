use thiserror::Error;

/// Errors aligned with Unix errno values.
#[derive(Debug, Error)]
pub enum FSError {
    /// ENOENT — no such file or directory.
    #[error("ENOENT: no such file or directory: {0}")]
    NotFound(String),

    /// EEXIST — file already exists.
    #[error("EEXIST: file already exists: {0}")]
    AlreadyExists(String),

    /// EISDIR — is a directory.
    #[error("EISDIR: is a directory: {0}")]
    IsADirectory(String),

    /// ENOTDIR — not a directory.
    #[error("ENOTDIR: not a directory: {0}")]
    NotADirectory(String),

    /// ENOTEMPTY — directory not empty.
    #[error("ENOTEMPTY: directory not empty: {0}")]
    DirectoryNotEmpty(String),

    /// EACCES — permission denied.
    #[error("EACCES: permission denied: {0}")]
    PermissionDenied(String),

    /// ELOOP — too many levels of symbolic links.
    #[error("ELOOP: too many levels of symbolic links: {0}")]
    TooManySymlinks(String),

    /// EINVAL — invalid argument (e.g. bad path component).
    #[error("EINVAL: invalid argument: {0}")]
    InvalidArgument(String),

    /// ENAMETOOLONG — file name too long.
    #[error("ENAMETOOLONG: file name too long: {0}")]
    NameTooLong(String),

    /// ENOTTY — not valid UTF-8 (no direct errno match; used for text-mode reads).
    #[error("not valid UTF-8: {0}")]
    NotUtf8(String),
}
