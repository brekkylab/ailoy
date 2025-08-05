mod cache;
mod manifest;

use std::pin::Pin;

pub use cache::*;

pub trait FromCache {
    fn from_cache(
        cache: Cache,
        key: impl AsRef<str>,
    ) -> Pin<Box<dyn Future<Output = Result<Self, String>>>>
    where
        Self: Sized;
}
