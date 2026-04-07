use std::collections::HashMap;

use crate::state::fs::{DirEntry, Directory, File, FileHandle, Node, NodeKind, Stat};

pub struct InMemoryFile {
    content: Vec<u8>,
    readonly: bool,
}

impl InMemoryFile {
    pub fn new() -> Self {
        Self {
            content: Vec::new(),
            readonly: false,
        }
    }

    pub fn with_content(content: Vec<u8>) -> Self {
        Self {
            content,
            readonly: false,
        }
    }

    /// Mark the file as read-only (builder-style).
    pub fn readonly(mut self) -> Self {
        self.readonly = true;
        self
    }
}

pub struct InMemoryFileHandle<'a> {
    source: &'a mut Vec<u8>,
    cursor: u64,
    readonly: bool,
}

impl File for InMemoryFile {
    type Handle<'a> = InMemoryFileHandle<'a>;

    fn open<'a>(&'a mut self) -> InMemoryFileHandle<'a> {
        InMemoryFileHandle {
            source: &mut self.content,
            cursor: 0,
            readonly: self.readonly,
        }
    }

    fn stat(&self) -> Stat {
        Stat {
            size: self.content.len() as u64,
            kind: NodeKind::File,
            readonly: self.readonly,
        }
    }
}

impl FileHandle for InMemoryFileHandle<'_> {
    fn read(&mut self, count: u64) -> &[u8] {
        let start = self.cursor as usize;
        let end = (self.cursor + count).min(self.source.len() as u64) as usize;
        self.cursor = end as u64;
        // self.cursor.set(end as u64);
        &self.source[start..end]
    }

    fn write(&mut self, data: &[u8]) {
        assert!(!self.readonly, "write to read-only file");
        let start = self.cursor as usize;
        let end = start + data.len();
        if end > self.source.len() {
            self.source.resize(end, 0);
        }
        self.source[start..end].copy_from_slice(data);
        self.cursor = end as u64;
    }

    fn seek(&mut self, offset: i64) -> u64 {
        let size = self.source.len() as i64;
        let new_cursor = (self.cursor as i64 + offset).clamp(0, size);
        self.cursor = new_cursor as u64;
        new_cursor as u64
    }

    fn tell(&mut self) -> u64 {
        self.cursor
    }
}

pub struct InMemoryDir {
    pub children: HashMap<String, Node>,
    readonly: bool,
}

impl InMemoryDir {
    pub fn new() -> Self {
        Self {
            children: HashMap::new(),
            readonly: false,
        }
    }

    /// Mark the directory as read-only (builder-style).
    pub fn readonly(mut self) -> Self {
        self.readonly = true;
        self
    }
}

impl Directory for InMemoryDir {
    fn readdir(&self) -> Vec<DirEntry> {
        self.children
            .iter()
            .map(|(name, node)| DirEntry {
                name: name.clone(),
                kind: node.kind(),
            })
            .collect()
    }

    fn stat(&self) -> Stat {
        Stat {
            size: self.children.len() as u64,
            kind: NodeKind::Directory,
            readonly: self.readonly,
        }
    }

    fn get_child(&self, name: &str) -> Option<&crate::state::fs::Node> {
        self.children.get(name)
    }

    fn get_child_mut(&mut self, name: &str) -> Option<&mut crate::state::fs::Node> {
        self.children.get_mut(name)
    }

    fn insert_child(&mut self, name: String, node: crate::state::fs::Node) {
        assert!(!self.readonly, "insert into read-only directory: {name}");
        self.children.insert(name, node);
    }

    fn remove_child(&mut self, name: &str) -> Option<Node> {
        assert!(!self.readonly, "remove from read-only directory: {name}");
        self.children.remove(name)
    }
}
