use std::any::Any;

use crate::{cache::CacheEntry, dyn_maybe_send};

pub struct CacheClaim {
    pub entries: Vec<CacheEntry>,
    pub ctx: Option<Box<dyn_maybe_send!(Any)>>,
}

impl CacheClaim {
    pub fn new(entries: impl IntoIterator<Item = impl Into<CacheEntry>>) -> Self {
        Self {
            entries: entries.into_iter().map(|v| v.into()).collect(),
            ctx: None,
        }
    }

    pub fn with_ctx(
        entries: impl IntoIterator<Item = impl Into<CacheEntry>>,
        ctx: Box<dyn_maybe_send!(Any)>,
    ) -> Self {
        Self {
            entries: entries.into_iter().map(|v| v.into()).collect(),
            ctx: Some(ctx),
        }
    }
}
