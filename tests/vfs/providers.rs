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

/// R2: the hot `ls -l` path. After `readdir` populates the cache, a `stat`
/// served by the cache fast-path must still report a real mtime — not the epoch.
/// Before R2 the cache `Entry` dropped mtime, so post-readdir stats showed 1970.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "live: needs AWS creds"]
async fn s3_readdir_then_stat_keeps_mtime() {
    dotenvy::dotenv().ok();
    if !has_mount("/s3") {
        eprintln!("no s3 creds — skipping");
        return;
    }
    let vfs = Vfs::from_config(all_vfs()).unwrap();
    let base = format!("vfs-r2-mtime-{}-{}", stamp(), uniq());
    let (res, fvp) = vfs.route(&format!("/s3/{base}/f.txt")).expect("route file");
    res.write_bytes(&fvp, b"hi".to_vec()).await.expect("write");

    // readdir the parent through the *same* (cached) resource so the cache is
    // populated — exactly what `ls` does before `ls -l` per-entry stats.
    let (res, dvp) = vfs.route(&format!("/s3/{base}")).expect("route dir");
    let _ = res.readdir(&dvp).await.expect("readdir fills cache");

    // Now the file stat is served by the cache fast-path. It must carry mtime.
    let (res, fvp) = vfs.route(&format!("/s3/{base}/f.txt")).expect("route file");
    let st = res.stat(&fvp).await.expect("stat after readdir");
    let mtime = st
        .mtime
        .expect("cached stat must keep a real mtime (R2), got None");
    assert!(
        mtime > std::time::UNIX_EPOCH + std::time::Duration::from_secs(946_684_800),
        "cached mtime should be a real (post-2000) time, not the epoch: {mtime:?}"
    );

    // Cleanup.
    let (res, dvp) = vfs.route(&format!("/s3/{base}")).expect("route dir");
    let _ = res.rmdir(&dvp).await;
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

/// Direct smoke test of the Gmail adapter: list labels, descend a label into its
/// date dirs, read an email `.gmail.json` (asserting the processed shape),
/// and best-effort read one attachment. A scope/enablement problem (the token
/// lacking Gmail access) is reported and treated as a soft skip — it's an
/// external-config issue, not an adapter bug.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "live: needs GOOGLE_* refresh token with a Gmail scope (gmail.modify)"]
async fn gmail_labels_dates_and_read_smoke() {
    dotenvy::dotenv().ok();
    if !has_mount("/gmail") {
        eprintln!("no google creds — skipping");
        return;
    }
    let vfs = Vfs::from_config(all_vfs()).unwrap();

    // Labels (root readdir). A 403 here means the refresh token has no Gmail
    // scope — surface it clearly and skip (not an adapter failure).
    let (res, vp) = vfs.route("/gmail").expect("route gmail");
    let labels = match res.readdir(&vp).await {
        Ok(l) => l,
        Err(e) => {
            eprintln!(
                "NOTE: gmail readdir failed — likely the refresh token lacks a \
                 Gmail scope (re-issue with gmail.modify): {e}"
            );
            return;
        }
    };
    let names: Vec<&str> = labels.iter().map(|e| e.name.as_str()).collect();
    println!("gmail labels: {names:?}");
    assert!(
        labels.iter().all(|e| e.kind == FileKind::Dir),
        "labels must be directories"
    );
    assert!(
        names.contains(&"INBOX"),
        "expected the INBOX system label, got {names:?}"
    );

    // Dates within INBOX.
    let (res, vp) = vfs.route("/gmail/INBOX").expect("route INBOX");
    let dates = res.readdir(&vp).await.expect("readdir INBOX");
    println!(
        "INBOX dates ({}, newest first): {:?}",
        dates.len(),
        dates.iter().take(5).map(|e| &e.name).collect::<Vec<_>>()
    );
    let Some(date) = dates.first() else {
        println!("NOTE: INBOX is empty — read/attachment checks skipped");
        return;
    };
    // Date dirs must be yyyy-mm-dd and sorted newest-first.
    assert!(
        date.name.len() == 10 && date.name.as_bytes()[4] == b'-',
        "date dir should be yyyy-mm-dd, got {}",
        date.name
    );

    // Messages within that date.
    let (res, vp) = vfs
        .route(&format!("/gmail/INBOX/{}", date.name))
        .expect("route date");
    let entries = res.readdir(&vp).await.expect("readdir date");
    let msg = entries
        .iter()
        .find(|e| e.name.ends_with(".gmail.json"))
        .expect("a date dir must contain at least one .gmail.json");
    println!("first message file: {}", msg.name);

    // Read the email JSON and assert the processed schema.
    let (res, vp) = vfs
        .route(&format!("/gmail/INBOX/{}/{}", date.name, msg.name))
        .expect("route msg");
    let data = res.read_bytes(&vp, None).await.expect("read .gmail.json");
    let email: serde_json::Value = serde_json::from_slice(&data).expect("valid email JSON");
    for k in [
        "id",
        "thread_id",
        "from",
        "to",
        "cc",
        "subject",
        "date",
        "body_text",
        "snippet",
        "labels",
        "attachments",
    ] {
        assert!(email.get(k).is_some(), ".gmail.json missing key `{k}`");
    }
    assert!(
        email.get("from").and_then(|f| f.get("email")).is_some(),
        "from must be {{name,email}}"
    );
    println!(
        "read email: from={:?} subject={:?}",
        email["from"]["email"], email["subject"]
    );

    // A ranged read must slice (direct_io reads aren't clamped to stat size).
    let head = res.read_bytes(&vp, Some(0..1)).await.expect("ranged read");
    assert_eq!(head, b"{", "email JSON should start with '{{'");

    // Best-effort: if this message has attachments, list + read the first.
    let atts = email["attachments"].as_array().cloned().unwrap_or_default();
    if !atts.is_empty() {
        let dir = msg.name.trim_end_matches(".gmail.json");
        let (res, vp) = vfs
            .route(&format!("/gmail/INBOX/{}/{}", date.name, dir))
            .expect("route attach dir");
        let files = res.readdir(&vp).await.expect("readdir attachments");
        assert!(!files.is_empty(), "attachment dir should list files");
        let f = &files[0];
        let (res, vp) = vfs
            .route(&format!("/gmail/INBOX/{}/{}/{}", date.name, dir, f.name))
            .expect("route attachment");
        let bytes = res.read_bytes(&vp, None).await.expect("read attachment");
        println!(
            "attachment {} -> {} bytes (stat size {})",
            f.name,
            bytes.len(),
            f.size
        );
        assert_eq!(
            bytes.len() as u64,
            f.size,
            "attachment bytes should match its stat size"
        );
    } else {
        println!("(message has no attachments — attachment read skipped)");
    }
}

/// Direct smoke test of the Gmail `.cmd/send` control write. Gated on
/// `GMAIL_TEST_TO` so it never emails a random address; sends a one-off message
/// and asserts the API returns a message id.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "live: sends a real email — needs GOOGLE_* (gmail.send/modify) + GMAIL_TEST_TO"]
async fn gmail_command_send_smoke() {
    dotenvy::dotenv().ok();
    if !has_mount("/gmail") {
        eprintln!("no google creds — skipping");
        return;
    }
    let Ok(to) = std::env::var("GMAIL_TEST_TO") else {
        eprintln!("set GMAIL_TEST_TO=you@example.com to exercise gmail send — skipping");
        return;
    };
    let vfs = Vfs::from_config(all_vfs()).unwrap();
    let (res, _) = vfs.route("/gmail/.cmd/send").expect("route .cmd/send");
    let body = serde_json::json!({
        "to": to,
        "subject": format!("ailoy vfs gmail smoke {}", stamp()),
        "body": "Sent by the ailoy vfs gmail adapter test.\n",
    });
    match res.command("send", body.to_string().as_bytes()).await {
        Ok(result) => {
            let v: serde_json::Value = serde_json::from_slice(&result).unwrap();
            let id = v.get("id").and_then(|x| x.as_str());
            assert!(id.is_some(), "send response should carry a message id: {v}");
            println!(
                "gmail send OK -> id={:?} thread={:?}",
                id,
                v.get("threadId")
            );
        }
        Err(e) => {
            println!(
                "NOTE: gmail send reached the API but failed (token gmail.send \
                 scope / enablement): {e}"
            );
        }
    }
}

/// Live test of the Gmail reply / reply-all / forward control writes. Seeds a
/// message to `GMAIL_TEST_TO` (use your own address), then drives each threaded
/// send off that seed's id and asserts the API returns an id — and that reply /
/// reply-all stay in the seed's thread. Sends real email; gated on GMAIL_TEST_TO.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "live: sends real emails — needs GOOGLE_* (gmail.modify) + GMAIL_TEST_TO"]
async fn gmail_reply_forward_smoke() {
    dotenvy::dotenv().ok();
    if !has_mount("/gmail") {
        eprintln!("no google creds — skipping");
        return;
    }
    let Ok(to) = std::env::var("GMAIL_TEST_TO") else {
        eprintln!("set GMAIL_TEST_TO=you@example.com to exercise reply/forward — skipping");
        return;
    };
    let vfs = Vfs::from_config(all_vfs()).unwrap();
    let (res, _) = vfs.route("/gmail/.cmd/send").expect("route gmail .cmd");

    let cmd = |op: &'static str, body: serde_json::Value| {
        let res = res.clone();
        async move {
            let out = res
                .command(op, body.to_string().as_bytes())
                .await
                .unwrap_or_else(|e| panic!("{op} failed: {e}"));
            serde_json::from_slice::<serde_json::Value>(&out).expect("json result")
        }
    };

    // Seed a message we own, so reply/forward have a real original to thread off.
    let seed = cmd(
        "send",
        serde_json::json!({
            "to": to,
            "subject": format!("ailoy reply/fwd seed {}", stamp()),
            "body": "seed for reply/forward smoke\n",
        }),
    )
    .await;
    let seed_id = seed["id"].as_str().expect("seed id").to_string();
    let seed_thread = seed["threadId"].as_str().map(String::from);
    println!("seed id={seed_id} thread={seed_thread:?}");

    // reply — must carry an id and stay in the seed's thread.
    let r = cmd(
        "reply",
        serde_json::json!({"message_id": seed_id, "body": "live reply\n"}),
    )
    .await;
    assert!(r["id"].as_str().is_some(), "reply must return an id: {r}");
    if let Some(t) = &seed_thread {
        assert_eq!(
            r["threadId"].as_str(),
            Some(t.as_str()),
            "reply must stay in-thread"
        );
    }
    println!("reply id={:?} thread={:?}", r["id"], r["threadId"]);

    // reply-all — same threading guarantee.
    let ra = cmd(
        "reply-all",
        serde_json::json!({"message_id": seed_id, "body": "live reply-all\n"}),
    )
    .await;
    assert!(
        ra["id"].as_str().is_some(),
        "reply-all must return an id: {ra}"
    );
    if let Some(t) = &seed_thread {
        assert_eq!(
            ra["threadId"].as_str(),
            Some(t.as_str()),
            "reply-all must stay in-thread"
        );
    }
    println!("reply-all id={:?} thread={:?}", ra["id"], ra["threadId"]);

    // forward — a fresh message (new thread) to the recipient.
    let f = cmd(
        "forward",
        serde_json::json!({"message_id": seed_id, "to": to}),
    )
    .await;
    assert!(f["id"].as_str().is_some(), "forward must return an id: {f}");
    println!("forward id={:?} thread={:?}", f["id"], f["threadId"]);
    println!("gmail reply / reply-all / forward OK ✅");
}

/// Regression: Gmail listings are capped (newest 50), so a date dir absent from
/// the label-level listing must still be reachable. Reproduces the host-mount
/// flow — list the parent (populating the cache), then `stat`/`readdir` an older
/// date — which previously hit the cache's negative-ENOENT shortcut and failed
/// with "No such file or directory". A far-past date keeps this independent of
/// the live mailbox's contents.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "live: needs GOOGLE_* with a Gmail scope (gmail.modify)"]
async fn gmail_capped_listing_does_not_hide_older_dates() {
    dotenvy::dotenv().ok();
    if !has_mount("/gmail") {
        eprintln!("no google creds — skipping");
        return;
    }
    let vfs = Vfs::from_config(all_vfs()).unwrap();

    // Populate the cache exactly like `ls /gmail/INBOX` does (capped listing).
    let (res, vp) = vfs.route("/gmail/INBOX").expect("route INBOX");
    if let Err(e) = res.readdir(&vp).await {
        eprintln!("NOTE: gmail readdir failed (token Gmail scope?): {e} — skipping");
        return;
    }

    // A date not in that capped listing must NOT be negative-cached as absent:
    // stat must resolve it as a directory (the mount's `lookup` step), and
    // readdir must succeed (date-narrowed query), not ENOENT.
    let (res, vp) = vfs
        .route("/gmail/INBOX/2020-01-01")
        .expect("route old date");
    let st = res
        .stat(&vp)
        .await
        .expect("stat of an older date dir must not be negative-cached to ENOENT");
    assert_eq!(st.kind, FileKind::Dir, "a valid date path is a directory");
    // The date-narrowed listing resolves (likely empty for a far-past day).
    let entries = res
        .readdir(&vp)
        .await
        .expect("readdir of an older date dir");
    println!(
        "/gmail/INBOX/2020-01-01 reachable after capped INBOX listing: {} entries",
        entries.len()
    );
}

/// Once an older date dir has been visited via `ls <label>/<date>` and turned
/// out to have mail, it should be folded into the (capped) label listing on the
/// next `ls <label>` — incrementally, and so tab-completion can offer it. Uses
/// the same Vfs instance so the cache is shared, mirroring a live shell session.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "live: needs GOOGLE_* with a Gmail scope (gmail.modify)"]
async fn gmail_visited_dates_fold_into_label_listing() {
    dotenvy::dotenv().ok();
    if !has_mount("/gmail") {
        eprintln!("no google creds — skipping");
        return;
    }
    let vfs = Vfs::from_config(all_vfs()).unwrap();
    let inbox_dates = || {
        let vfs = &vfs;
        async move {
            let (res, vp) = vfs.route("/gmail/INBOX").unwrap();
            res.readdir(&vp)
                .await
                .map(|e| e.into_iter().map(|d| d.name).collect::<Vec<_>>())
        }
    };

    let before = match inbox_dates().await {
        Ok(d) => d,
        Err(e) => {
            eprintln!("NOTE: gmail readdir failed (token Gmail scope?): {e} — skipping");
            return;
        }
    };
    println!("INBOX before: {before:?}");

    // Find an older date (not already shown) that has mail, and visit it.
    let candidates = [
        "2026-06-01",
        "2026-05-15",
        "2026-05-01",
        "2026-04-15",
        "2026-04-01",
        "2026-03-01",
    ];
    let mut visited: Option<String> = None;
    for d in candidates {
        if before.iter().any(|x| x == d) {
            continue;
        }
        let (res, vp) = vfs.route(&format!("/gmail/INBOX/{d}")).unwrap();
        let msgs = res
            .readdir(&vp)
            .await
            .map(|e| e.iter().filter(|x| x.name.ends_with(".gmail.json")).count())
            .unwrap_or(0);
        if msgs > 0 {
            println!("visited older date {d} -> {msgs} message(s)");
            visited = Some(d.to_string());
            break;
        }
    }
    let Some(d) = visited else {
        println!("no older date with mail among candidates — skipping incremental assert");
        return;
    };

    // Re-list INBOX: the visited older date must now be folded in.
    let after = inbox_dates().await.expect("readdir INBOX again");
    println!("INBOX after: {after:?}");
    assert!(
        after.contains(&d),
        "visited older date {d} should now appear in the INBOX listing: {after:?}"
    );
    assert!(
        after.len() > before.len() || before.contains(&d),
        "listing should have grown to include {d}"
    );
}
