use crate::state::{Node, Shell};

use super::ExecOutput;

pub fn run_rmdir(shell: &mut Shell, args: &[&str]) -> ExecOutput {
    let mut remove_parents = false;
    let mut paths: Vec<&str> = Vec::new();

    for &arg in args {
        if arg == "-p" {
            remove_parents = true;
        } else if arg.starts_with('-') {
            return ExecOutput::err(
                1,
                format!("rmdir: illegal option -- {}", arg.trim_start_matches('-')),
            );
        } else {
            paths.push(arg);
        }
    }

    if paths.is_empty() {
        return ExecOutput::err(1, "rmdir: missing operand");
    }

    for &path in &paths {
        let mut components = shell.resolve(path);
        loop {
            let Some((name, parent_components)) = components.split_last() else {
                return ExecOutput::err(
                    1,
                    "rmdir: failed to remove '/': Device or resource busy",
                );
            };
            let name = name.clone();
            let parent_components = parent_components.to_vec();

            let parent_ptr =
                match Shell::navigate_mut(&mut shell.root, &parent_components) {
                    Ok(p) => p,
                    Err(_) => {
                        return ExecOutput::err(
                            1,
                            format!(
                                "rmdir: failed to remove '{path}': No such file or directory"
                            ),
                        )
                    }
                };
            // SAFETY: ptr points to a live Directory; no aliasing after navigate_mut returns.
            let parent = unsafe { &mut *parent_ptr };
            match parent.get_child(name.as_str()) {
                None => {
                    return ExecOutput::err(
                        1,
                        format!("rmdir: failed to remove '{path}': No such file or directory"),
                    )
                }
                Some(Node::File(_)) => {
                    return ExecOutput::err(
                        1,
                        format!("rmdir: failed to remove '{path}': Not a directory"),
                    )
                }
                Some(Node::Directory(d)) if !d.readdir().is_empty() => {
                    return ExecOutput::err(
                        1,
                        format!("rmdir: failed to remove '{path}': Directory not empty"),
                    )
                }
                Some(Node::Directory(_)) => {}
            }
            parent.remove_child(name.as_str());

            if !remove_parents {
                break;
            }
            components = parent_components;
            if components.is_empty() {
                break;
            }
        }
    }

    ExecOutput::ok("")
}
