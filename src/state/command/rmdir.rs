use crate::state::{Node, Shell};

pub fn run_rmdir(shell: &mut Shell, args: &[&str]) -> anyhow::Result<String> {
    let mut remove_parents = false;
    let mut paths: Vec<&str> = Vec::new();

    for &arg in args {
        if arg == "-p" {
            remove_parents = true;
        } else if arg.starts_with('-') {
            anyhow::bail!("rmdir: illegal option -- {}", arg.trim_start_matches('-'));
        } else {
            paths.push(arg);
        }
    }

    if paths.is_empty() {
        anyhow::bail!("rmdir: missing operand");
    }

    for &path in &paths {
        let mut components = shell.resolve(path);
        loop {
            let Some((name, parent_components)) = components.split_last() else {
                anyhow::bail!("rmdir: failed to remove '/': Device or resource busy");
            };
            let name = name.clone();
            let parent_components = parent_components.to_vec();

            let parent_ptr =
                Shell::navigate_mut(&mut shell.root, &parent_components).map_err(|_| {
                    anyhow::anyhow!("rmdir: failed to remove '{path}': No such file or directory")
                })?;
            // SAFETY: ptr points to a live Directory; no aliasing after navigate_mut returns.
            let parent = unsafe { &mut *parent_ptr };
            match parent.get_child(name.as_str()) {
                None => {
                    anyhow::bail!("rmdir: failed to remove '{path}': No such file or directory")
                }
                Some(Node::File(_)) => {
                    anyhow::bail!("rmdir: failed to remove '{path}': Not a directory")
                }
                Some(Node::Directory(d)) if !d.readdir().is_empty() => {
                    anyhow::bail!("rmdir: failed to remove '{path}': Directory not empty")
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

    Ok(String::new())
}
