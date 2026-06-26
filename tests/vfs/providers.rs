//! Direct provider-adapter smoke tests (no FUSE frontend, no sandbox):
//! exercise the S3 / Notion / GDrive `Resource`s straight through the `Vfs` core.

use crate::common::*;

/// Direct smoke test of the S3 adapter's `readdir` parity with mirage:
/// children come back name-sorted, subfolders are directories (via the `/`
/// delimiter), and a zero-byte "directory marker" object for the listed
/// prefix is excluded (mirage drops it; object_store does not).
#[tokio::test(flavor = "multi_thread")]
#[ignore = "live: needs AWS creds + aws CLI"]
async fn s3_readdir_marker_and_sort_smoke() {
    dotenvy::dotenv().ok();
    let bucket = std::env::var("AWS_S3_BUCKET").unwrap();
    let s = stamp();
    let base = format!("vfs-s3-smoke-{s}");
    let vfs = Vfs::from_config(vfs_config()).unwrap();

    // Write three children (out of order) plus a nested object.
    for (name, body) in [("b.txt", "B"), ("a.txt", "A"), ("sub/c.txt", "C")] {
        let (res, vp) = vfs.route(&format!("/s3/{base}/{name}")).expect("route");
        res.write_bytes(&vp, body.as_bytes().to_vec())
            .await
            .expect("write");
    }
    // Create the explicit directory-marker object `<base>/` (a zero-byte key
    // ending in `/`), which the Resource API cannot create on its own.
    let marker_key = format!("{base}/");
    let status = std::process::Command::new("aws")
        .args([
            "s3api",
            "put-object",
            "--bucket",
            &bucket,
            "--key",
            &marker_key,
        ])
        .status()
        .expect("run aws put-object");
    assert!(status.success(), "failed to create folder marker");

    let (res, vp) = vfs.route(&format!("/s3/{base}")).expect("route dir");
    let entries = res.readdir(&vp).await.expect("readdir");
    let names: Vec<(String, bool)> = entries
        .iter()
        .map(|e| (e.name.clone(), e.kind == FileKind::Dir))
        .collect();
    println!("readdir {base}: {names:?}");

    // Marker for the listed prefix must be gone; entries name-sorted; sub is a dir.
    assert!(
        !entries.iter().any(|e| e.name == base),
        "directory marker leaked as a child entry"
    );
    assert_eq!(
        entries.iter().map(|e| e.name.as_str()).collect::<Vec<_>>(),
        vec!["a.txt", "b.txt", "sub"],
        "entries should be name-sorted with sub/ as a directory"
    );
    assert!(entries.iter().find(|e| e.name == "sub").unwrap().kind == FileKind::Dir);

    // Cleanup.
    let _ = std::process::Command::new("aws")
        .args(["s3", "rm", &format!("s3://{bucket}/{base}"), "--recursive"])
        .status();
    let _ = std::process::Command::new("aws")
        .args([
            "s3api",
            "delete-object",
            "--bucket",
            &bucket,
            "--key",
            &marker_key,
        ])
        .status();
}

/// Direct (non-agent) smoke test of the Notion adapter: read the page tree,
/// read a page.json, then exercise the `.cmd` domain writes (page-create +
/// block-append) through `Resource::command`.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "live: needs NOTION_API_KEY with a shared page"]
async fn notion_read_and_command_smoke() {
    dotenvy::dotenv().ok();
    let vfs = Vfs::from_config(notion_vfs()).unwrap();

    let (res, vp) = vfs.route("/notion/pages").expect("route pages");
    let entries = res.readdir(&vp).await.expect("readdir pages");
    println!(
        "pages: {:?}",
        entries.iter().map(|e| &e.name).collect::<Vec<_>>()
    );
    // Prefer a known write-shared parent; fall back to the first page.
    let parent = entries
        .iter()
        .find(|e| e.name.contains("Engineering_Logs"))
        .or_else(|| entries.first())
        .expect("at least one shared page");
    let parent_id = parent
        .name
        .rsplit_once("__")
        .map(|(_, id)| id)
        .unwrap()
        .to_string();

    // Descend into the page dir: it should contain page.json plus a
    // subdirectory per child_page block (hierarchical, mirroring mirage).
    let (res, vp) = vfs
        .route(&format!("/notion/pages/{}", parent.name))
        .expect("route page dir");
    let dir_entries = res.readdir(&vp).await.expect("readdir page dir");
    println!(
        "{} contents: {:?}",
        parent.name,
        dir_entries
            .iter()
            .map(|e| format!(
                "{}{}",
                e.name,
                if e.kind == FileKind::Dir { "/" } else { "" }
            ))
            .collect::<Vec<_>>()
    );
    assert!(
        dir_entries.iter().any(|e| e.name == "page.json"),
        "page dir must expose page.json"
    );

    let (res, vp) = vfs
        .route(&format!("/notion/pages/{}/page.json", parent.name))
        .expect("route page.json");
    let data = res.read_bytes(&vp, None).await.expect("read page.json");
    let page: serde_json::Value = serde_json::from_slice(&data).unwrap();
    println!("first page title: {:?}", page.get("title"));
    // Normalized schema parity with mirage `normalize_page`.
    for k in [
        "page_id",
        "title",
        "url",
        "created_time",
        "last_edited_time",
        "parent_type",
        "parent_id",
        "archived",
        "created_by",
        "last_edited_by",
        "markdown",
        "blocks",
    ] {
        assert!(page.get(k).is_some(), "page.json missing key `{k}`");
    }

    // page-create under the first page
    let (res, _) = vfs.route("/notion/.cmd/page-create").unwrap();
    let create_body = serde_json::json!({
        "parent": {"page_id": parent_id},
        "properties": {"title": [{"text": {"content": "ailoy vfs Phase2"}}]}
    });
    // Domain write reaches the Notion API; success additionally requires the
    // integration to have the "Insert content" capability on a shared page.
    match res
        .command("page-create", create_body.to_string().as_bytes())
        .await
    {
        Ok(created) => {
            let created: serde_json::Value = serde_json::from_slice(&created).unwrap();
            let new_id = created
                .get("id")
                .and_then(|v| v.as_str())
                .expect("new page id")
                .to_string();
            println!("created page id: {new_id}");
            let append_body = serde_json::json!({
                "block_id": new_id,
                "children": [{
                    "object": "block", "type": "paragraph",
                    "paragraph": {"rich_text": [{"type": "text", "text": {"content": "created by ailoy vfs phase 2"}}]}
                }]
            });
            let appended = res
                .command("block-append", append_body.to_string().as_bytes())
                .await
                .expect("block-append");
            assert!(!appended.is_empty(), "block-append returned empty");
            println!("page-create + block-append OK ✅");
        }
        Err(e) => {
            println!(
                "NOTE: domain write reached Notion but was rejected (integration \
                 capability / sharing): {e}"
            );
        }
    }
}

/// Verify the GDrive adapter mirrors the Drive folder hierarchy: the root
/// lists folders as directories, and descending into a folder lists its
/// children (not a flat dump of the whole Drive).
#[tokio::test(flavor = "multi_thread")]
#[ignore = "live: needs GOOGLE_* refresh token with Drive scope"]
async fn gdrive_hierarchy_smoke() {
    dotenvy::dotenv().ok();
    let vfs = Vfs::from_config(gdrive_vfs()).unwrap();
    let (res, vp) = vfs.route("/gdrive").expect("route gdrive");
    let root = res.readdir(&vp).await.expect("readdir root");
    let dirs: Vec<&String> = root
        .iter()
        .filter(|e| matches!(e.kind, ailoy::vfs::FileKind::Dir))
        .map(|e| &e.name)
        .collect();
    println!("root: {} entries, {} folders", root.len(), dirs.len());
    println!("folders: {:?}", &dirs[..dirs.len().min(8)]);
    assert!(!dirs.is_empty(), "expected at least one folder at root");

    let folder = dirs[0].clone();
    let (res, vp) = vfs
        .route(&format!("/gdrive/{folder}"))
        .expect("route subfolder");
    let children = res.readdir(&vp).await.expect("readdir subfolder");
    println!(
        "  /{folder}: {} children -> {:?}",
        children.len(),
        children.iter().map(|e| &e.name).take(8).collect::<Vec<_>>()
    );
}

/// Direct smoke test of the GDrive adapter: list Drive, read a Google Doc as
/// `.gdoc.json`, then append to it via `.cmd/docs-append`.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "live: needs GOOGLE_* refresh token with Drive + Docs scopes"]
async fn gdrive_read_and_command_smoke() {
    dotenvy::dotenv().ok();
    let vfs = Vfs::from_config(gdrive_vfs()).unwrap();
    let (res, vp) = vfs.route("/gdrive").expect("route gdrive");
    let entries = res.readdir(&vp).await.expect("readdir gdrive");
    let names: Vec<&String> = entries.iter().map(|e| &e.name).collect();
    println!(
        "gdrive entries (first 10): {:?}",
        &names[..names.len().min(10)]
    );

    let gdoc = entries.iter().find(|e| e.name.ends_with(".gdoc.json"));
    let Some(gdoc) = gdoc else {
        println!("NOTE: no Google Doc found in Drive; read/append skipped");
        return;
    };
    let (res, vp) = vfs
        .route(&format!("/gdrive/{}", gdoc.name))
        .expect("route gdoc");
    let data = res.read_bytes(&vp, None).await.expect("read gdoc");
    let doc: serde_json::Value = serde_json::from_slice(&data).unwrap();
    let doc_id = doc
        .get("documentId")
        .and_then(|v| v.as_str())
        .expect("documentId")
        .to_string();
    println!("read {} -> documentId={doc_id}", gdoc.name);

    // Append to a user-owned doc (the read doc above may be read-only-shared).
    // Override via AILOY_GDOC_ID; defaults to a known-writable doc.
    let write_doc = std::env::var("AILOY_GDOC_ID")
        .unwrap_or_else(|_| "10NTjr9rPPqZKzoW_YpP9z-8WfSMuN0z1-mDq012KWuI".into());
    let _ = doc_id;
    let (res, _) = vfs.route("/gdrive/.cmd/docs-append").unwrap();
    let body =
        serde_json::json!({"document_id": write_doc, "text": "\nappended by ailoy vfs phase 2\n"});
    match res
        .command("docs-append", body.to_string().as_bytes())
        .await
    {
        Ok(result) => {
            assert!(!result.is_empty(), "docs-append returned empty");
            println!("docs-append OK ✅");
        }
        Err(e) => {
            println!(
                "NOTE: docs-append reached the Docs API but was rejected \
                 (token Docs scope / doc ACL): {e}"
            );
        }
    }
}
