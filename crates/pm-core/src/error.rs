//! Error type shared by the core crate.

/// Errors produced by `pm-core`.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// An underlying I/O operation failed.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// An image could not be decoded or converted.
    #[error("image error: {0}")]
    Image(String),

    /// A model or result file is malformed or has an unsupported version.
    #[error("format error: {0}")]
    Format(String),

    /// A caller supplied an invalid parameter.
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),

    /// A numerical routine failed to produce a usable result.
    #[error("numerical failure: {0}")]
    Numerical(String),

    /// The caller cancelled a long-running operation through its progress hook.
    #[error("cancelled")]
    Cancelled,
}

/// Convenience alias for results returned by `pm-core`.
pub type Result<T> = std::result::Result<T, Error>;
