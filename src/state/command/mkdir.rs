use crate::state::{InMemoryDir, Node, Shell};

pub fn run_mkdir(shell: &mut Shell, args: &[&str]) -> anyhow::Result<String> {
    let mut make_parents = false;
    let mut paths: Vec<&str> = Vec::new();

    for &arg in args {
        if arg == "-p" {
            make_parents = true;
        } else if arg.starts_with('-') {
            anyhow::bail!("mkdir: illegal option -- {}", arg.trim_start_matches('-'));
        } else {
            paths.push(arg);
        }
    }

    if paths.is_empty() {
        anyhow::bail!("mkdir: missing operand");
    }

    for &path in &paths {
        let components = shell.resolve(path);
        if make_parents {
            for i in 0..components.len() {
                let prefix = &components[..=i];
                let (name, parent_components) = prefix.split_last().unwrap();
                let parent_ptr =
                    Shell::navigate_mut(&mut shell.root, parent_components).map_err(|_| {
                        anyhow::anyhow!(
                            "mkdir: cannot create directory '{path}': No such file or directory"
                        )
                    })?;
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
                        anyhow::bail!("mkdir: cannot create directory '{path}': Not a directory");
                    }
                }
            }
        } else {
            let Some((name, parent_components)) = components.split_last() else {
                anyhow::bail!("mkdir: cannot create directory '/'");
            };
            let parent_ptr =
                Shell::navigate_mut(&mut shell.root, parent_components).map_err(|_| {
                    anyhow::anyhow!(
                        "mkdir: cannot create directory '{path}': No such file or directory"
                    )
                })?;
            // SAFETY: ptr points to a live Directory; no aliasing after navigate_mut returns.
            let parent = unsafe { &mut *parent_ptr };
            if parent.get_child(name.as_str()).is_some() {
                anyhow::bail!("mkdir: cannot create directory '{path}': File exists");
            }
            parent.insert_child(name.clone(), Node::Directory(Box::new(InMemoryDir::new())));
        }
    }

    Ok(String::new())
}
