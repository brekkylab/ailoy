use crate::shell::{InMemoryDir, Node, Shell};

use super::ExecOutput;

pub fn run_mkdir(shell: &mut Shell, args: &[&str]) -> ExecOutput {
    let mut make_parents = false;
    let mut paths: Vec<&str> = Vec::new();

    for &arg in args {
        if arg == "-p" {
            make_parents = true;
        } else if arg.starts_with('-') {
            return ExecOutput::err(
                1,
                format!("mkdir: illegal option -- {}", arg.trim_start_matches('-')),
            );
        } else {
            paths.push(arg);
        }
    }

    if paths.is_empty() {
        return ExecOutput::err(1, "mkdir: missing operand");
    }

    for &path in &paths {
        let components = shell.resolve(path);
        if make_parents {
            for i in 0..components.len() {
                let prefix = &components[..=i];
                let (name, parent_components) = prefix.split_last().unwrap();
                let parent_ptr = match Shell::navigate_mut(&mut shell.root, parent_components) {
                    Ok(p) => p,
                    Err(_) => {
                        return ExecOutput::err(
                            1,
                            format!(
                                "mkdir: cannot create directory '{path}': No such file or directory"
                            ),
                        );
                    }
                };
                // SAFETY: ptr points to a live Directory; no aliasing after navigate_mut returns.
                let parent = unsafe { &mut *parent_ptr };
                match parent.get_child(name.as_str()) {
                    None => {
                        parent.insert_child(
                            name.clone(),
                            Node::Directory(Box::new(InMemoryDir::new())),
                        );
                    }
                    Some(Node::Directory(_)) => {}
                    Some(Node::File(_)) => {
                        return ExecOutput::err(
                            1,
                            format!("mkdir: cannot create directory '{path}': Not a directory"),
                        );
                    }
                }
            }
        } else {
            let Some((name, parent_components)) = components.split_last() else {
                return ExecOutput::err(1, "mkdir: cannot create directory '/'");
            };
            let parent_ptr = match Shell::navigate_mut(&mut shell.root, parent_components) {
                Ok(p) => p,
                Err(_) => {
                    return ExecOutput::err(
                        1,
                        format!(
                            "mkdir: cannot create directory '{path}': No such file or directory"
                        ),
                    );
                }
            };
            // SAFETY: ptr points to a live Directory; no aliasing after navigate_mut returns.
            let parent = unsafe { &mut *parent_ptr };
            if parent.get_child(name.as_str()).is_some() {
                return ExecOutput::err(
                    1,
                    format!("mkdir: cannot create directory '{path}': File exists"),
                );
            }
            parent.insert_child(name.clone(), Node::Directory(Box::new(InMemoryDir::new())));
        }
    }

    ExecOutput::ok("")
}
