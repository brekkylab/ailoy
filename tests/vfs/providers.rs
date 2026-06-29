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
    if !has_mount("/s3") {
        eprintln!("no s3 creds — skipping");
        return;
    }
    let bucket = std::env::var("AWS_S3_BUCKET").unwrap();
    let s = stamp();
    let base = format!("vfs-s3-smoke-{s}");
    let vfs = Vfs::from_config(all_vfs()).unwrap();

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

/// Direct S3 stat metadata (S3-1) + range-past-EOF handling (S3-2).
#[tokio::test(flavor = "multi_thread")]
#[ignore = "live: needs AWS creds"]
async fn s3_stat_metadata_and_range_smoke() {
    dotenvy::dotenv().ok();
    if !has_mount("/s3") {
        eprintln!("no s3 creds — skipping");
        return;
    }
    let vfs = Vfs::from_config(all_vfs()).unwrap();
    let (res, vp) = vfs
        .route(&format!("/s3/vfs-meta-{}-{}.txt", stamp(), uniq()))
        .expect("route");
    res.write_bytes(&vp, b"hello".to_vec())
        .await
        .expect("write");

    // S3-1: stat reports size + a real mtime + etag.
    let st = res.stat(&vp).await.expect("stat");
    assert_eq!(st.size, 5);
    assert!(st.mtime.is_some(), "S3 stat should report mtime");
    assert!(st.etag.is_some(), "S3 stat should report etag");

    // S3-2: a range starting at/after EOF returns empty (clean EOF), not an error.
    let past = res
        .read_bytes(&vp, Some(5..100))
        .await
        .expect("range at EOF should be Ok(empty), not an error");
    assert!(
        past.is_empty(),
        "range at EOF should be empty, got {past:?}"
    );
    // A range that starts in-bounds but overruns is clamped to the available bytes.
    let tail = res
        .read_bytes(&vp, Some(3..100))
        .await
        .expect("overrun range");
    assert_eq!(
        tail, b"lo",
        "overrunning range should clamp to available bytes"
    );
    // A fully in-bounds range slices exactly.
    let mid = res
        .read_bytes(&vp, Some(1..3))
        .await
        .expect("in-bounds range");
    assert_eq!(mid, b"el");

    let _ = res.unlink(&vp).await;
}

/// Direct S3 directory ops (C3): rename (copy+delete), recursive rmdir, mkdir.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "live: needs AWS creds"]
async fn s3_dir_ops_smoke() {
    dotenvy::dotenv().ok();
    if !has_mount("/s3") {
        eprintln!("no s3 creds — skipping");
        return;
    }
    let vfs = Vfs::from_config(all_vfs()).unwrap();
    let base = format!("vfs-dirops-{}-{}", stamp(), uniq());

    // mkdir is a no-op success on S3 (implicit dirs); it must not error.
    let (res, dvp) = vfs.route(&format!("/s3/{base}")).expect("route dir");
    res.mkdir(&dvp).await.expect("mkdir should succeed (no-op)");

    // Write base/a, rename to base/b: b has the content, a is gone.
    let (res, a) = vfs.route(&format!("/s3/{base}/a")).expect("route a");
    res.write_bytes(&a, b"DATA".to_vec())
        .await
        .expect("write a");
    let (res, b) = vfs.route(&format!("/s3/{base}/b")).expect("route b");
    res.rename(&a, &b).await.expect("rename a->b");
    assert_eq!(res.read_bytes(&b, None).await.expect("read b"), b"DATA");
    assert!(
        res.read_bytes(&a, None).await.is_err(),
        "source should be gone after rename"
    );

    // rmdir removes the whole prefix recursively.
    let (res, dvp) = vfs.route(&format!("/s3/{base}")).expect("route dir");
    res.rmdir(&dvp).await.expect("rmdir");
    let (res, b) = vfs.route(&format!("/s3/{base}/b")).expect("route b");
    assert!(
        res.read_bytes(&b, None).await.is_err(),
        "rmdir should remove contents"
    );
}

/// N1: stat of a nonexistent Notion page errors (ENOENT-able) instead of being
/// reported as a directory.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "live: needs NOTION_API_KEY"]
async fn notion_stat_nonexistent() {
    dotenvy::dotenv().ok();
    if !has_mount("/notion") {
        eprintln!("no notion creds — skipping");
        return;
    }
    let vfs = Vfs::from_config(all_vfs()).unwrap();
    let (res, vp) = vfs
        .route("/notion/pages/Bogus__00000000000000000000000000000000")
        .expect("route");
    assert!(
        res.stat(&vp).await.is_err(),
        "stat of a nonexistent notion page should error, not report a directory"
    );
}

/// Direct (non-agent) smoke test of the Notion adapter: read the page tree,
/// read a page.json, then exercise the `.cmd` domain writes (page-create +
/// block-append) through `Resource::command`.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "live: needs NOTION_API_KEY with a shared page"]
async fn notion_read_and_command_smoke() {
    dotenvy::dotenv().ok();
    if !has_mount("/notion") {
        eprintln!("no notion creds — skipping");
        return;
    }
    let vfs = Vfs::from_config(all_vfs()).unwrap();

    let (res, vp) = vfs.route("/notion/pages").expect("route pages");
    let entries = res.readdir(&vp).await.expect("readdir pages");
    println!(
        "pages: {:?}",
        entries.iter().map(|e| &e.name).collect::<Vec<_>>()
    );
    // Use whatever page is shared with the integration; assume no specific one.
    let Some(parent) = entries.first() else {
        eprintln!("no notion pages shared with the integration — skipping");
        return;
    };
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

/// Multi-mount of the SAME provider type with isolated config — the plan's
/// first-class "N instances of one provider, each with its own creds/scope".
/// Mounts S3 twice: `/s3` (whole bucket) and `/scoped` (a `key_prefix`). Proves
/// (a) top-level prefix routing reaches the right instance, and (b) `key_prefix`
/// is a real scope: a write through `/scoped/<f>` lands under the prefix, so the
/// unprefixed mount sees it only at `/s3/<prefix>/<f>`, never at `/s3/<f>`.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "live: multi-mount same provider, key_prefix isolation (needs AWS creds)"]
async fn s3_multi_mount_key_prefix_isolation() {
    dotenvy::dotenv().ok();
    if !has_mount("/s3") {
        eprintln!("no s3 creds — skipping");
        return;
    }
    let s = format!("{}-{}", stamp(), uniq());
    let key_prefix = format!("vfs-isolated-{s}");
    let s3 = |prefix: Option<String>| S3Config {
        bucket: std::env::var("AWS_S3_BUCKET").unwrap(),
        region: std::env::var("AWS_DEFAULT_REGION").unwrap_or_else(|_| "us-east-1".into()),
        access_key_id: std::env::var("AWS_ACCESS_KEY_ID").unwrap(),
        secret_access_key: std::env::var("AWS_SECRET_ACCESS_KEY").unwrap(),
        endpoint: None,
        key_prefix: prefix,
    };
    let vfs = Vfs::from_config(VfsConfig {
        mounts: vec![
            MountSpec {
                prefix: "/s3".into(),
                provider: ProviderConfig::S3(s3(None)),
            },
            MountSpec {
                prefix: "/scoped".into(),
                provider: ProviderConfig::S3(s3(Some(key_prefix.clone()))),
            },
        ],
    })
    .unwrap();

    let fname = format!("iso-{s}.txt");
    let content = format!("isolated-{s}");

    // Write through the scoped (key_prefix'd) mount, read it back there.
    let (res, vp) = vfs
        .route(&format!("/scoped/{fname}"))
        .expect("route scoped");
    res.write_bytes(&vp, content.clone().into_bytes())
        .await
        .expect("write scoped");
    let (res, vp) = vfs
        .route(&format!("/scoped/{fname}"))
        .expect("route scoped read");
    let got = res.read_bytes(&vp, None).await.expect("read scoped");
    assert_eq!(
        String::from_utf8_lossy(&got),
        content,
        "scoped mount should read back its own write"
    );

    // The unprefixed mount must NOT see it at the bare path (it's under the prefix)…
    let (res, vp) = vfs.route(&format!("/s3/{fname}")).expect("route base bare");
    assert!(
        res.read_bytes(&vp, None).await.is_err(),
        "base mount must not see the scoped object at the unprefixed path"
    );
    // …but DOES see it under the prefix, proving key_prefix is a real scope.
    let (res, vp) = vfs
        .route(&format!("/s3/{key_prefix}/{fname}"))
        .expect("route base prefixed");
    let got2 = res
        .read_bytes(&vp, None)
        .await
        .expect("base mount should see the object at the prefixed key");
    assert_eq!(String::from_utf8_lossy(&got2), content);

    // Cleanup via the scoped mount.
    let (res, vp) = vfs
        .route(&format!("/scoped/{fname}"))
        .expect("route scoped unlink");
    let _ = res.unlink(&vp).await;
    println!("S3 MULTI-MOUNT KEY-PREFIX ISOLATION OK");
}

/// Verify the GDrive adapter mirrors the Drive folder hierarchy: the root
/// lists folders as directories, and descending into a folder lists its
/// children (not a flat dump of the whole Drive).
#[tokio::test(flavor = "multi_thread")]
#[ignore = "live: needs GOOGLE_* refresh token with Drive scope"]
async fn gdrive_hierarchy_smoke() {
    dotenvy::dotenv().ok();
    if !has_mount("/gdrive") {
        eprintln!("no gdrive creds — skipping");
        return;
    }
    let vfs = Vfs::from_config(all_vfs()).unwrap();
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
    if !has_mount("/gdrive") {
        eprintln!("no gdrive creds — skipping");
        return;
    }
    let vfs = Vfs::from_config(all_vfs()).unwrap();
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

    // Append to the doc we just discovered (no hardcoded doc id). It may be
    // read-only-shared, in which case the Docs API rejects the write and we treat
    // that as a soft skip in the `Err` arm below.
    let (res, _) = vfs.route("/gdrive/.cmd/docs-append").unwrap();
    let body =
        serde_json::json!({"document_id": doc_id, "text": "\nappended by ailoy vfs phase 2\n"});
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
