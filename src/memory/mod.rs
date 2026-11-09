//! What an agent remembers: a [`MemoryFile`] it writes notes in on this host.

mod file;

pub(crate) use file::used;
pub use file::{EditError, EditRefusal, MemoryEdit, MemoryFile};
