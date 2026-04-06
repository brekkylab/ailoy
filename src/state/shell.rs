use crate::state::command::ls::run_ls;
use crate::state::command::mkdir::run_mkdir;
use crate::state::command::rmdir::run_rmdir;

use super::fs::{Directory, InMemoryDir, Node};

pub struct Shell {
    pub(super) cwd: Vec<String>,
    pub(crate) root: InMemoryDir,
}

impl Shell {
    pub fn new() -> Self {
        let mut root = InMemoryDir::new();
        let mut home = InMemoryDir::new();
        home.insert_child("user".into(), Node::Directory(Box::new(InMemoryDir::new())));
        root.insert_child("home".into(), Node::Directory(Box::new(home)));
        Self {
            cwd: vec!["home".into(), "user".into()],
            root,
        }
    }

    /// Execute a shell command
    pub fn exec(&mut self, cmd: &str) -> anyhow::Result<String> {
        let parts: Vec<&str> = cmd.split_whitespace().collect();
        match parts.as_slice() {
            ["mkdir", args @ ..] => run_mkdir(self, args),
            ["rmdir", args @ ..] => run_rmdir(self, args),
            ["ls", args @ ..] => run_ls(self, args),
            [] => anyhow::bail!("empty command"),
            _ => anyhow::bail!("unknown command: {}", cmd),
        }
    }

    // ─── Path helpers ─────────────────────────────────────────────────────

    /// Resolve `path` against `cwd` and return normalized path components
    /// (no leading slash, no `.` or `..`).
    pub(crate) fn resolve(&self, path: &str) -> Vec<String> {
        let mut components = if path.starts_with('/') {
            Vec::new()
        } else {
            self.cwd.clone()
        };

        for part in path.split('/') {
            match part {
                "" | "." => {}
                ".." => {
                    components.pop();
                }
                p => components.push(p.to_string()),
            }
        }
        components
    }

    pub(crate) fn navigate<'a>(
        root: &'a dyn Directory,
        components: &[String],
    ) -> anyhow::Result<*const (dyn Directory + 'a)> {
        let mut ptr: *const dyn Directory = root;
        for name in components {
            // SAFETY: ptr always points to a live Directory in the tree.
            let current = unsafe { &*ptr };
            match current.get_child(name.as_str()) {
                Some(Node::Directory(d)) => ptr = d.as_ref(),
                Some(Node::File(_)) => anyhow::bail!("not a directory: {name}"),
                None => anyhow::bail!("no such file or directory: {name}"),
            }
        }
        Ok(ptr)
    }

    pub(crate) fn navigate_mut<'a>(
        root: &'a mut dyn Directory,
        components: &[String],
    ) -> anyhow::Result<*mut (dyn Directory + 'a)> {
        let mut ptr: *mut dyn Directory = root;
        for name in components {
            // SAFETY: ptr always points to a live Directory in the tree.
            let current = unsafe { &mut *ptr };
            match current.get_child_mut(name.as_str()) {
                Some(Node::Directory(d)) => ptr = d.as_mut(),
                Some(Node::File(_)) => anyhow::bail!("not a directory: {name}"),
                None => anyhow::bail!("no such file or directory: {name}"),
            }
        }
        Ok(ptr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ls_mkdir_ls_rmdir_ls() {
        let mut shell = Shell::new();

        // ls — cwd is empty
        assert_eq!(shell.exec("ls").unwrap(), "");

        // mkdir testdir1
        shell.exec("mkdir testdir1").unwrap();

        // mkdir testdir2
        shell.exec("mkdir testdir2").unwrap();

        // ls
        assert_eq!(shell.exec("ls").unwrap(), "testdir1  testdir2");

        // rmdir testdir1
        shell.exec("rmdir testdir1").unwrap();

        // ls
        assert_eq!(shell.exec("ls").unwrap(), "testdir2");

        // rmdir testdir2
        shell.exec("rmdir testdir2").unwrap();

        // ls
        assert_eq!(shell.exec("ls").unwrap(), "");
    }
}
