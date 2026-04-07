use crate::state::Shell;

use super::ExecOutput;

pub fn run_mv(shell: &mut Shell, args: &[&str]) -> ExecOutput {
    let mut paths: Vec<&str> = Vec::new();

    for &arg in args {
        if arg.starts_with('-') {
            return ExecOutput::err(
                1,
                format!("mv: invalid option -- '{}'", arg.trim_start_matches('-')),
            );
        } else {
            paths.push(arg);
        }
    }

    if paths.is_empty() {
        return ExecOutput::err(1, "mv: missing file operand");
    }
    if paths.len() < 2 {
        return ExecOutput::err(
            1,
            format!("mv: missing destination file operand after '{}'", paths[0]),
        );
    }

    let dst_str = paths[paths.len() - 1];
    let srcs = &paths[..paths.len() - 1];
    let dst_components = shell.resolve(dst_str);

    // navigate() succeeds iff the full path resolves to an existing directory.
    let dst_is_dir = Shell::navigate(&shell.root, &dst_components).is_ok();

    if srcs.len() > 1 && !dst_is_dir {
        return ExecOutput::err(1, format!("mv: target '{dst_str}' is not a directory"));
    }

    for &src_str in srcs {
        let src_components = shell.resolve(src_str);
        let Some((src_name, src_parent_components)) = src_components.split_last() else {
            return ExecOutput::err(1, "mv: cannot move '/'");
        };
        let src_name = src_name.clone();
        let src_parent_components = src_parent_components.to_vec();

        // Take the source node from its parent.
        let src_parent_ptr =
            match Shell::navigate_mut(&mut shell.root, &src_parent_components) {
                Ok(p) => p,
                Err(_) => {
                    return ExecOutput::err(
                        1,
                        format!("mv: cannot stat '{src_str}': No such file or directory"),
                    )
                }
            };
        // SAFETY: ptr is valid and we hold a mutable borrow of shell for this block.
        let src_parent = unsafe { &mut *src_parent_ptr };
        let node = match src_parent.remove_child(&src_name) {
            Some(n) => n,
            None => {
                return ExecOutput::err(
                    1,
                    format!("mv: cannot stat '{src_str}': No such file or directory"),
                )
            }
        };
        // src_parent borrow ends here; raw pointer remains valid.

        // Determine destination name and its parent path components.
        let (dst_name, dst_parent_components) = if dst_is_dir {
            (src_name, dst_components.clone())
        } else {
            match dst_components.split_last() {
                Some((name, parent)) => (name.clone(), parent.to_vec()),
                None => return ExecOutput::err(1, "mv: cannot move to '/'"),
            }
        };

        let dst_parent_ptr =
            match Shell::navigate_mut(&mut shell.root, &dst_parent_components) {
                Ok(p) => p,
                Err(_) => {
                    return ExecOutput::err(
                        1,
                        format!(
                            "mv: cannot move '{src_str}' to '{dst_str}': No such file or directory"
                        ),
                    )
                }
            };
        // SAFETY: ptr is valid and no other live mutable reference aliases it.
        let dst_parent = unsafe { &mut *dst_parent_ptr };
        dst_parent.insert_child(dst_name, node);
    }

    ExecOutput::ok("")
}
