use crate::state::fs::Node;
use crate::state::Shell;

use super::ExecOutput;

pub fn run_cat(shell: &mut Shell, args: &[&str]) -> ExecOutput {
    if args.is_empty() {
        // No stdin support in virtual shell — behave like blocking read (output nothing).
        return ExecOutput::ok("");
    }

    let mut stdout_parts: Vec<String> = Vec::new();
    let mut stderr_lines: Vec<String> = Vec::new();
    let mut exit_code = 0;

    for &path_str in args {
        if path_str.starts_with('-') {
            stderr_lines.push(format!(
                "cat: {path_str}: No such file or directory"
            ));
            exit_code = 1;
            continue;
        }

        let components = shell.resolve(path_str);
        let Some((name, parent_components)) = components.split_last() else {
            stderr_lines.push(format!("cat: /: Is a directory"));
            exit_code = 1;
            continue;
        };
        let name = name.clone();
        let parent_components = parent_components.to_vec();

        let parent_ptr = match Shell::navigate_mut(&mut shell.root, &parent_components) {
            Ok(p) => p,
            Err(_) => {
                stderr_lines.push(format!(
                    "cat: {path_str}: No such file or directory"
                ));
                exit_code = 1;
                continue;
            }
        };
        // SAFETY: ptr is valid and we hold a mutable borrow of shell.
        let parent = unsafe { &mut *parent_ptr };

        match parent.get_child_mut(&name) {
            None => {
                stderr_lines.push(format!("cat: {path_str}: No such file or directory"));
                exit_code = 1;
            }
            Some(Node::Directory(_)) => {
                stderr_lines.push(format!("cat: {path_str}: Is a directory"));
                exit_code = 1;
            }
            Some(Node::File(f)) => {
                let size = f.stat_dyn().size;
                let mut handle = f.open_dyn();
                let bytes = handle.read(size).to_vec();
                drop(handle);
                match String::from_utf8(bytes) {
                    Ok(s) => stdout_parts.push(s),
                    Err(e) => {
                        // Output the lossy UTF-8 representation, same as cat on binary files.
                        stdout_parts.push(String::from_utf8_lossy(e.as_bytes()).into_owned());
                    }
                }
            }
        }
    }

    let stdout = stdout_parts.join("");
    let stderr = stderr_lines.join("\n");
    ExecOutput { stdout, stderr, exit_code }
}
