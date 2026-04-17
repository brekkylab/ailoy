fn main() {
    if std::env::var("CARGO_FEATURE_SANDBOX_KRUN").is_err() {
        return;
    }

    // Find onlybots-runner produced by onlybots' build.rs and copy it into
    // ailoy's target directory so that `find_runner_binary()` can locate it.
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let profile = std::env::var("PROFILE").unwrap();

    // onlybots workspace is a sibling of ailoy: ../onlybots
    let onlybots_runner = std::path::Path::new(&manifest_dir)
        .parent()
        .unwrap()
        .join("onlybots")
        .join("target")
        .join(&profile)
        .join("onlybots-runner");

    if !onlybots_runner.exists() {
        println!(
            "cargo:warning=onlybots-runner not found at {:?}; build it first with: \
             cd ../onlybots && cargo build",
            onlybots_runner
        );
        return;
    }

    // OUT_DIR is target/<profile>/build/ailoy-<hash>/out — walk up to target/<profile>
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let target_dir = std::path::Path::new(&out_dir)
        .parent().unwrap() // ailoy-<hash>
        .parent().unwrap() // build
        .parent().unwrap(); // target/<profile>

    let dst = target_dir.join("onlybots-runner");
    if let Err(e) = std::fs::copy(&onlybots_runner, &dst) {
        println!("cargo:warning=failed to copy onlybots-runner to {:?}: {}", dst, e);
    } else {
        println!("cargo:rerun-if-changed={}", onlybots_runner.display());
    }
}
