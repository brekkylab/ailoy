mod cache;
mod compat;
mod fs;
mod manifest;

pub use cache::*;

pub trait CacheUse {
    fn on_initialize(&self, cache: &Cache);
}
