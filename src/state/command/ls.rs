use crate::state::fs::{DirEntry, Directory, Node, NodeKind};
use crate::state::Shell;

pub fn run_ls(shell: &Shell, args: &[&str]) -> anyhow::Result<String> {
    let mut show_all = false;
    let mut long_format = false;
    let mut path = "";

    for &arg in args {
        if arg.starts_with('-') {
            for ch in arg[1..].chars() {
                match ch {
                    'a' => show_all = true,
                    'l' => long_format = true,
                    _ => anyhow::bail!("ls: illegal option -- {ch}"),
                }
            }
        } else {
            path = arg;
        }
    }

    let components = shell.resolve(path);
    let dir_ptr = Shell::navigate(&shell.root as &dyn Directory, &components)
        .map_err(|_| anyhow::anyhow!("ls: cannot access '{path}': No such file or directory"))?;

    // SAFETY: ptr is valid and we hold a shared borrow of shell.
    let dir = unsafe { &*dir_ptr };
    let mut entries = dir.readdir();

    if !show_all {
        entries.retain(|e| !e.name.starts_with('.'));
    }

    entries.sort_by(|a, b| a.name.cmp(&b.name));

    if show_all {
        entries.insert(0, DirEntry { name: "..".to_string(), kind: NodeKind::Directory });
        entries.insert(0, DirEntry { name: ".".to_string(), kind: NodeKind::Directory });
    }

    if entries.is_empty() {
        return Ok(String::new());
    }

    if long_format {
        let lines: Vec<String> = entries
            .iter()
            .map(|e| {
                let type_char = match e.kind {
                    NodeKind::Directory => 'd',
                    NodeKind::File => '-',
                };
                let perm = match e.kind {
                    NodeKind::Directory => "rwxr-xr-x",
                    NodeKind::File => "rw-r--r--",
                };
                let size: u64 = match e.name.as_str() {
                    "." => dir.stat().size,
                    ".." => 0,
                    name => match dir.get_child(name) {
                        Some(Node::File(f)) => f.stat_dyn().size,
                        Some(Node::Directory(d)) => d.stat().size,
                        None => 0,
                    },
                };
                format!("{type_char}{perm}  {:>6}  {}", size, e.name)
            })
            .collect();
        Ok(lines.join("\n"))
    } else {
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
