use crate::shell::Shell;
use crate::shell::fs::Node;

use super::ExecOutput;

pub fn run_rm(shell: &mut Shell, args: &[&str]) -> ExecOutput {
    let mut recursive = false;
    let mut force = false;
    let mut paths: Vec<&str> = Vec::new();

    for &arg in args {
        if arg.starts_with('-') && arg.len() > 1 {
            for ch in arg[1..].chars() {
                match ch {
                    'r' | 'R' => recursive = true,
                    'f' => force = true,
                    _ => return ExecOutput::err(1, format!("rm: invalid option -- '{ch}'")),
                }
            }
        } else {
            paths.push(arg);
        }
    }

    if paths.is_empty() && !force {
        return ExecOutput::err(1, "rm: missing operand");
    }

    let mut exit_code = 0;
    let mut stderr_lines: Vec<String> = Vec::new();

    for &path_str in &paths {
        let components = shell.resolve(path_str);

        if components.is_empty() {
            stderr_lines.push(
                "rm: it is dangerous to operate recursively on '/'\n\
                 rm: use --no-preserve-root to override this failsafe"
                    .to_string(),
            );
            exit_code = 1;
            continue;
        }

        let Some((name, parent_components)) = components.split_last() else {
            unreachable!()
        };
        let name = name.clone();
        let parent_components = parent_components.to_vec();

        let parent_ptr = match Shell::navigate_mut(&mut shell.root, &parent_components) {
            Ok(p) => p,
            Err(_) => {
                if !force {
                    stderr_lines.push(format!(
                        "rm: cannot remove '{path_str}': No such file or directory"
                    ));
                    exit_code = 1;
                }
                continue;
            }
        };
        // SAFETY: ptr is valid and we hold a mutable borrow of shell.
        let parent = unsafe { &mut *parent_ptr };

        match parent.get_child(&name) {
            None => {
                if !force {
                    stderr_lines.push(format!(
                        "rm: cannot remove '{path_str}': No such file or directory"
                    ));
                    exit_code = 1;
                }
            }
            Some(Node::Directory(_)) if !recursive => {
                stderr_lines.push(format!("rm: cannot remove '{path_str}': Is a directory"));
                exit_code = 1;
            }
            _ => {
                parent.remove_child(&name);
            }
        }
    }

    if stderr_lines.is_empty() {
        ExecOutput::ok("")
    } else {
        ExecOutput {
            stdout: String::new(),
            stderr: stderr_lines.join("\n"),
            exit_code,
        }
    }
}
