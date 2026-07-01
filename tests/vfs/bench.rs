//! Host-vs-guest overhead benchmarks (manual, non-CI).

use crate::common::*;

/// Quantify where the guest (sandbox forward) path spends time vs the host
/// (direct in-process) path. Brings up the forward server + a sandbox, mounts
/// the forwarder, then measures from inside the guest:
///   - pure transport RTT (`/stat?path=/` short-circuits before any provider),
///   - one `readdir`,
///   - a serial `stat` of every entry (what `ls -la` triggers),
/// and compares against the same Resource calls in-process on the host.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "bench: measures guest-forward vs host overhead (needs AWS + sandbox + macFUSE)"]
async fn vfs_forward_overhead_bench() {
    use std::{sync::Arc, time::Instant};

    dotenvy::dotenv().ok();

    // S3-only so provider latency is uniform across entries.
    let cfg = VfsConfig {
        mounts: vec![MountSpec {
            prefix: "/s3".into(),
            provider: ProviderConfig::S3(S3Config {
                bucket: std::env::var("AWS_S3_BUCKET").unwrap(),
                region: std::env::var("AWS_DEFAULT_REGION").unwrap_or_else(|_| "us-east-1".into()),
                access_key_id: std::env::var("AWS_ACCESS_KEY_ID").unwrap(),
                secret_access_key: std::env::var("AWS_SECRET_ACCESS_KEY").unwrap(),
                endpoint: None,
                key_prefix: None,
            }),
        }],
    };
    let vfs = Arc::new(Vfs::from_config(cfg).unwrap());
    let rt = tokio::runtime::Handle::current();
    let forward = ailoy::vfs::VfsForward::spawn(vfs.clone(), &rt).expect("forward");
    let port = forward.port();
    let token = forward.token().to_string();

    // ---- Host baseline (in-process) ----
    let (res, vp) = vfs.route("/s3").expect("route");
    let t = Instant::now();
    let entries = res.readdir(&vp).await.expect("host readdir");
    let host_readdir_ms = t.elapsed().as_secs_f64() * 1e3;
    let n = entries.len();
    let t = Instant::now();
    for e in &entries {
        let (r, p) = vfs.route(&format!("/s3/{}", e.name)).unwrap();
        let _ = r.stat(&p).await;
    }
    let host_stat_all_ms = t.elapsed().as_secs_f64() * 1e3;
    // Concurrent — this is how the real host FUSE mount (multithreaded macFUSE)
    // actually issues getattr, unlike the guest forwarder's nothreads=True.
    let t = Instant::now();
    let futs = entries.iter().map(|e| {
        let (r, p) = vfs.route(&format!("/s3/{}", e.name)).unwrap();
        async move { r.stat(&p).await }
    });
    let _ = futures::future::join_all(futs).await;
    let host_stat_concurrent_ms = t.elapsed().as_secs_f64() * 1e3;

    // ---- Bring up the sandbox + in-guest forwarder ----
    let sandbox = RunEnv::sandbox(SandboxConfig {
        ..Default::default()
    })
    .await
    .expect("sandbox");
    let handle = sandbox.get().await.expect("handle");
    ailoy::vfs::bootstrap_guest_forwarder(&handle, "/mnt/vfs", port, &token)
        .await
        .expect("bootstrap forwarder");

    // Bench script runs entirely in the guest; isolates transport from FUSE.
    let bench = r#"
import os, time, urllib.request, urllib.parse
BASE = os.environ["B"]; TOK = os.environ["T"]; MNT = "/mnt/vfs/s3"
def req(route, path):
    r = urllib.request.Request(BASE + route + "?" + urllib.parse.urlencode({"path": path}), method="GET")
    r.add_header("x-vfs-token", TOK)
    return urllib.request.urlopen(r, timeout=60).read()
N = 50
t = time.perf_counter()
for _ in range(N): req("/stat", "/")
print("transport_per_req_ms", round((time.perf_counter()-t)/N*1000, 2))
t = time.perf_counter(); names = os.listdir(MNT); ld = (time.perf_counter()-t)*1000
print("guest_listdir_ms", round(ld, 2), "n", len(names))
t = time.perf_counter()
for nm in names:
    try: os.stat(os.path.join(MNT, nm))
    except OSError: pass
sa = (time.perf_counter()-t)*1000
print("guest_stat_all_ms", round(sa, 2))
print("guest_per_entry_stat_ms", round(sa/max(len(names),1), 2))
"#;
    let script = format!(
        "export B='http://host.microsandbox.internal:{port}'; export T='{token}'; \
         python3 -c \"$BENCH\"",
    );
    let script = format!("BENCH={}\n{script}", shell_quote(bench));
    let out = handle
        .exec_shell(script, Some(180))
        .await
        .expect("guest bench");

    println!("\n================ VFS FORWARD OVERHEAD ================");
    println!("entries (n)              : {n}");
    println!("HOST  readdir            : {host_readdir_ms:.1} ms");
    println!(
        "HOST  stat all (serial)  : {host_stat_all_ms:.1} ms  ({:.2} ms/entry)",
        host_stat_all_ms / n.max(1) as f64
    );
    println!("HOST  stat all (concurrent, ~real FUSE): {host_stat_concurrent_ms:.1} ms");
    println!("--- guest (sandbox forward) ---\n{}", out.stdout.trim());
    if !out.stderr.trim().is_empty() {
        println!("[guest stderr] {}", out.stderr.trim());
    }
    println!("=====================================================\n");
}

/// Single-quote a string for safe embedding in a POSIX shell assignment.
fn shell_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\\''"))
}

/// Per-provider host-vs-guest comparison, focused on Notion/GDrive where the
/// per-entry `stat` is itself expensive (Notion renders the page; GDrive
/// exports Workspace docs) — so the guest's serial-vs-host's-concurrent gap is
/// felt most. Measures, for a directory in each provider:
///   - host `readdir` + concurrent `stat` of every entry (≈ real host FUSE),
///   - guest `listdir` + serial `stat` of every entry (≈ `ls -la`),
/// using the updated forwarder (readdir-populated attr cache + multithreading).
#[tokio::test(flavor = "multi_thread")]
#[ignore = "bench: host vs guest per provider (needs all creds + sandbox + macFUSE)"]
async fn vfs_provider_overhead_bench() {
    use std::{sync::Arc, time::Instant};

    dotenvy::dotenv().ok();

    let vfs = Arc::new(Vfs::from_config(all_vfs()).unwrap());
    let rt = tokio::runtime::Handle::current();
    let forward = ailoy::vfs::VfsForward::spawn(vfs.clone(), &rt).expect("forward");
    let port = forward.port();
    let token = forward.token().to_string();

    // Pick a representative directory per provider (a Notion page dir whose
    // entries' stat triggers page rendering; the GDrive + S3 roots).
    let notion_dir = {
        let (res, vp) = vfs.route("/notion/pages").expect("route notion");
        let pages = res.readdir(&vp).await.expect("notion readdir");
        pages
            .first()
            .map(|e| format!("/notion/pages/{}", e.name))
            .expect("at least one notion page")
    };
    let targets = [
        ("notion", notion_dir.as_str()),
        ("gdrive", "/gdrive"),
        ("s3", "/s3"),
    ];

    // ---- Host baseline (in-process), readdir + concurrent stat ----
    let mut host_rows = Vec::new();
    for (label, path) in targets {
        let (res, vp) = vfs.route(path).unwrap();
        let t = Instant::now();
        let entries = res.readdir(&vp).await.expect("host readdir");
        let rd = t.elapsed().as_secs_f64() * 1e3;
        let n = entries.len();
        let t = Instant::now();
        let futs = entries.iter().map(|e| {
            let (r, p) = vfs.route(&format!("{path}/{}", e.name)).unwrap();
            async move { r.stat(&p).await }
        });
        let _ = futures::future::join_all(futs).await;
        let sa = t.elapsed().as_secs_f64() * 1e3;
        host_rows.push((label, n, rd, sa));
    }

    // ---- Bring up sandbox + new in-guest forwarder ----
    let sandbox = RunEnv::sandbox(SandboxConfig {
        ..Default::default()
    })
    .await
    .expect("sandbox");
    let handle = sandbox.get().await.expect("handle");
    ailoy::vfs::bootstrap_guest_forwarder(&handle, "/mnt/vfs", port, &token)
        .await
        .expect("bootstrap forwarder");

    // Guest: listdir + serial stat of every entry (what `ls -la` does), per dir.
    let bench = r#"
import os, time, json
for d in os.environ["DIRS"].split(","):
    g = "/mnt/vfs" + d
    t = time.perf_counter(); names = os.listdir(g); ld = (time.perf_counter()-t)*1000
    t = time.perf_counter()
    for nm in names:
        try: os.stat(os.path.join(g, nm))
        except OSError: pass
    sa = (time.perf_counter()-t)*1000
    print(json.dumps({"dir": d, "n": len(names),
                      "listdir_ms": round(ld,1), "stat_all_ms": round(sa,1)}))
"#;
    let dirs = targets
        .iter()
        .map(|(_, p)| *p)
        .collect::<Vec<_>>()
        .join(",");
    let script = format!(
        "export DIRS={}; BENCH={}\npython3 -c \"$BENCH\"",
        shell_quote(&dirs),
        shell_quote(bench),
    );
    let out = handle
        .exec_shell(script, Some(180))
        .await
        .expect("guest bench");

    println!("\n=================== HOST vs GUEST (per provider) ===================");
    println!(
        "{:<8} {:>3}  {:>16}  {:>22}",
        "provider", "n", "HOST rd+stat(conc)", "GUEST listdir+stat(ser)"
    );
    let guest: std::collections::HashMap<String, serde_json::Value> = out
        .stdout
        .lines()
        .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
        .filter_map(|v| {
            let dir = v.get("dir").and_then(|d| d.as_str())?.to_string();
            Some((dir, v))
        })
        .collect();
    for (label, n, rd, sa) in &host_rows {
        let path = targets.iter().find(|(l, _)| l == label).unwrap().1;
        let (gld, gsa) = guest
            .get(path)
            .map(|v| {
                (
                    v.get("listdir_ms").and_then(|x| x.as_f64()).unwrap_or(0.0),
                    v.get("stat_all_ms").and_then(|x| x.as_f64()).unwrap_or(0.0),
                )
            })
            .unwrap_or((0.0, 0.0));
        println!(
            "{label:<8} {n:>3}  {:>7.0}+{:>7.0}  {:>10.0}+{:>10.0}",
            rd, sa, gld, gsa
        );
    }
    if !out.stderr.trim().is_empty() {
        println!("[guest stderr] {}", out.stderr.trim());
    }
    println!("(ms; host stat is concurrent, guest stat is serial after a readdir)");
    println!("====================================================================\n");
}
