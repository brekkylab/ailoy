use crate::shell::Shell;
use crate::shell::fs::{Directory, InMemoryDir, InMemoryFile, Node};

use super::ExecOutput;

fn clone_node(node: &mut Node) -> Node {
    match node {
        Node::File(f) => {
            let size = f.stat_dyn().size;
            let mut handle = f.open_dyn();
            let content = handle.read(size).to_vec();
            drop(handle);
            Node::File(Box::new(InMemoryFile::with_content(content)))
        }
        Node::Directory(d) => {
            let names: Vec<String> = d.readdir().into_iter().map(|e| e.name).collect();
            let mut new_dir = InMemoryDir::new();
            for name in &names {
                if let Some(child) = d.get_child_mut(name) {
                    let cloned = clone_node(child);
                    new_dir.insert_child(name.clone(), cloned);
                }
            }
            Node::Directory(Box::new(new_dir))
        }
    }
}

pub fn run_cp(shell: &mut Shell, args: &[&str]) -> ExecOutput {
    let mut recursive = false;
    let mut paths: Vec<&str> = Vec::new();

    for &arg in args {
        if arg == "-r" || arg == "-R" || arg == "--recursive" {
            recursive = true;
        } else if arg.starts_with('-') {
            return ExecOutput::err(
                1,
                format!("cp: invalid option -- '{}'", arg.trim_start_matches('-')),
            );
        } else {
            paths.push(arg);
        }
    }

    if paths.is_empty() {
        return ExecOutput::err(1, "cp: missing file operand");
    }
    if paths.len() < 2 {
        return ExecOutput::err(
            1,
            format!("cp: missing destination file operand after '{}'", paths[0]),
        );
    }

    let dst_str = paths[paths.len() - 1];
    let srcs = &paths[..paths.len() - 1];
    let dst_components = shell.resolve(dst_str);

    // navigate() succeeds iff the full path resolves to an existing directory.
    let dst_is_dir = Shell::navigate(&shell.root, &dst_components).is_ok();

    if srcs.len() > 1 && !dst_is_dir {
        return ExecOutput::err(1, format!("cp: target '{dst_str}' is not a directory"));
    }

    for &src_str in srcs {
        let src_components = shell.resolve(src_str);
        let Some((src_name, src_parent_components)) = src_components.split_last() else {
            return ExecOutput::err(1, "cp: cannot copy '/'");
        };
        let src_name = src_name.clone();
        let src_parent_components = src_parent_components.to_vec();

        // Navigate to source parent and clone the node.
        let src_parent_ptr = match Shell::navigate_mut(&mut shell.root, &src_parent_components) {
            Ok(p) => p,
            Err(_) => {
                return ExecOutput::err(
                    1,
                    format!("cp: cannot stat '{src_str}': No such file or directory"),
                );
            }
        };
        // SAFETY: ptr is valid and we hold a mutable borrow of shell for this block.
        let src_parent = unsafe { &mut *src_parent_ptr };

        let src_node = match src_parent.get_child_mut(&src_name) {
            Some(n) => n,
            None => {
                return ExecOutput::err(
                    1,
                    format!("cp: cannot stat '{src_str}': No such file or directory"),
                );
            }
        };

        if matches!(src_node, Node::Directory(_)) && !recursive {
            return ExecOutput::err(1, format!("cp: omitting directory '{src_str}'"));
        }

        let cloned = clone_node(src_node);
        // src_parent borrow ends here (NLL); raw pointer remains valid.

        // Determine destination name and its parent path components.
        let (dst_name, dst_parent_components) = if dst_is_dir {
            (src_name, dst_components.clone())
        } else {
            match dst_components.split_last() {
                Some((name, parent)) => (name.clone(), parent.to_vec()),
                None => return ExecOutput::err(1, "cp: cannot copy to '/'"),
            }
        };

        let dst_parent_ptr = match Shell::navigate_mut(&mut shell.root, &dst_parent_components) {
            Ok(p) => p,
            Err(_) => {
                return ExecOutput::err(
                    1,
                    format!(
                        "cp: cannot create regular file '{dst_str}': No such file or directory"
                    ),
                );
            }
        };
        // SAFETY: ptr is valid and no other live mutable reference aliases it.
        let dst_parent = unsafe { &mut *dst_parent_ptr };
        dst_parent.insert_child(dst_name, cloned);
    }

    ExecOutput::ok("")
}
