//! What an agent remembers: a [`Memory`], a folder on this host whose `MEMORY.md` it writes
//! notes in.

mod file;

pub use file::{EditError, EditRefusal, FILE_NAME, Memory, MemoryEdit};
#[cfg(test)]
pub(crate) use file::today;
pub(crate) use file::used;
