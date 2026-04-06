use super::fs::{Directory, InMemoryDir, Node};

pub struct Shell {
    pub(super) cwd: Vec<String>,
    pub(super) root: InMemoryDir,
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
            ["mkdir", path] => self.cmd_mkdir(path),
            ["rmdir", path] => self.cmd_rmdir(path),
            ["ls"] => self.cmd_ls(""),
            ["ls", path] => self.cmd_ls(path),
            [] => anyhow::bail!("empty command"),
            _ => anyhow::bail!("unknown command: {}", cmd),
        }
    }

    // ─── Path helpers ─────────────────────────────────────────────────────

    /// Resolve `path` against `cwd` and return normalized path components
    /// (no leading slash, no `.` or `..`).
    fn resolve(&self, path: &str) -> Vec<String> {
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

    fn navigate<'a>(
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

    fn navigate_mut<'a>(
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

    // ─── Commands ─────────────────────────────────────────────────────────

    fn cmd_mkdir(&mut self, path: &str) -> anyhow::Result<String> {
        let components = self.resolve(path);
        let Some((name, parent_components)) = components.split_last() else {
            anyhow::bail!("mkdir: cannot create directory '/'");
        };

        let parent_ptr = Self::navigate_mut(&mut self.root, parent_components)
            .map_err(|_| anyhow::anyhow!("mkdir: {path}: No such file or directory"))?;

        // SAFETY: ptr points to a live Directory; no aliasing after navigate_mut returns.
        let parent = unsafe { &mut *parent_ptr };
        if parent.get_child(name.as_str()).is_some() {
            anyhow::bail!("mkdir: {path}: File exists");
        }
        parent.insert_child(name.clone(), Node::Directory(Box::new(InMemoryDir::new())));
        Ok(String::new())
    }

    fn cmd_rmdir(&mut self, path: &str) -> anyhow::Result<String> {
        let components = self.resolve(path);
        let Some((name, parent_components)) = components.split_last() else {
            anyhow::bail!("rmdir: failed to remove '/': Device or resource busy");
        };

        let parent_ptr = Self::navigate_mut(&mut self.root, parent_components)
            .map_err(|_| anyhow::anyhow!("rmdir: {path}: No such file or directory"))?;

        // SAFETY: same as cmd_mkdir.
        let parent = unsafe { &mut *parent_ptr };
        match parent.get_child(name.as_str()) {
            None => anyhow::bail!("rmdir: {path}: No such file or directory"),
            Some(Node::File(_)) => anyhow::bail!("rmdir: {path}: Not a directory"),
            Some(Node::Directory(d)) if !d.readdir().is_empty() => {
                anyhow::bail!("rmdir: {path}: Directory not empty")
            }
            Some(Node::Directory(_)) => {}
        }
        parent.remove_child(name.as_str());
        Ok(String::new())
    }

    fn cmd_ls(&self, path: &str) -> anyhow::Result<String> {
        let components = self.resolve(path);
        let dir_ptr = Self::navigate(&self.root as &dyn Directory, &components).map_err(|_| {
            anyhow::anyhow!("ls: cannot access '{path}': No such file or directory")
        })?;

        // SAFETY: ptr is valid and we hold a shared borrow of self.
        let dir = unsafe { &*dir_ptr };
        let mut entries = dir.readdir();
        entries.sort_by(|a, b| a.name.cmp(&b.name));

        if entries.is_empty() {
            return Ok(String::new());
        }

        let names: Vec<&str> = entries.iter().map(|e| e.name.as_str()).collect();
        let max_len = names.iter().map(|n| n.len()).max().unwrap_or(0);
        let col_width = max_len + 2;
        let terminal_width: usize = 80;
        let num_cols = (terminal_width / col_width).max(1);
        let num_rows = (names.len() + num_cols - 1) / num_cols;

        let mut rows: Vec<String> = Vec::with_capacity(num_rows);
        for row in 0..num_rows {
            let mut line = String::new();
            for col in 0..num_cols {
                let idx = col * num_rows + row;
                if idx >= names.len() {
                    break;
                }
                let last_in_row = (col + 1) * num_rows + row >= names.len();
                if last_in_row {
                    line.push_str(names[idx]);
                } else {
                    line.push_str(&format!("{:<col_width$}", names[idx]));
                }
            }
            rows.push(line);
        }

        Ok(rows.join("\n"))
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
