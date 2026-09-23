//! A real console on the local server cortex carries — nothing to build or install first.

use ailoy_desktop_core::{console::ConsoleFactory, workspace::WorkspaceMount};

/// The three trees a run is given, and what each one is for.
///
/// This is the arrangement the whole desktop rests on, and every part of it fails quietly if
/// the roles are swapped: the user's workspace goes in as the *context*, which cortex refuses
/// to let the agent write; what the agent produces goes in its *artifacts*; and the session
/// stands in its *scratch*, so a relative path is a throwaway one. Getting context and
/// artifacts the wrong way round still starts a console — it fails at the first write, which
/// is a whole run away from here.
#[tokio::test]
async fn the_three_trees_have_the_access_each_is_meant_to() {
    let workspace = tempfile::tempdir().unwrap();
    let artifacts = tempfile::tempdir().unwrap();
    let scratch = tempfile::tempdir().unwrap();
    std::fs::write(workspace.path().join("theirs.txt"), b"the user's").unwrap();

    let home = tempfile::tempdir().unwrap();
    let factory = ConsoleFactory::new(home.path().to_path_buf());
    let mut console = factory
        .spawn(
            WorkspaceMount(workspace.path().to_path_buf()),
            WorkspaceMount(artifacts.path().to_path_buf()),
            WorkspaceMount(scratch.path().to_path_buf()),
        )
        .await
        .unwrap();

    // Where it stands: a relative path is the scratch, not the user's files.
    let out = console.exec(["pwd"], Some(5_000)).await.unwrap();
    let cwd = String::from_utf8_lossy(&out.stdout).trim().to_string();
    assert_eq!(
        std::fs::canonicalize(&cwd).unwrap(),
        std::fs::canonicalize(scratch.path()).unwrap(),
        "the session should start in its scratch"
    );

    // The workspace is readable.
    let theirs = workspace.path().join("theirs.txt");
    let out = console
        .exec(["cat", &theirs.display().to_string()], Some(5_000))
        .await
        .unwrap();
    assert_eq!(
        out.code,
        0,
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(out.stdout, b"the user's".to_vec());

    // And not writable: the user's tree is managed outside the agent's life.
    let refused = console
        .write(
            workspace.path().join("mine.txt").display().to_string(),
            b"no".to_vec(),
            None,
        )
        .await;
    assert!(
        refused.is_err(),
        "a write into the context should be refused"
    );
    assert!(
        !workspace.path().join("mine.txt").exists(),
        "the refused write must not have landed"
    );

    // The artifacts tree is where the agent's own files go, and it takes the write.
    let mine = artifacts.path().join("mine.txt");
    console
        .write(
            mine.display().to_string(),
            b"from the session".to_vec(),
            None,
        )
        .await
        .expect("the artifacts tree takes a write");
    assert_eq!(std::fs::read_to_string(&mine).unwrap(), "from the session");
}
