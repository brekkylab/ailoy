//! Fetch the model catalog a release build embeds, into `assets/models.json`.
//!
//! `npm run catalog` in `apps/desktop` runs this, and `npm run tauri:build` runs that first.
//! The file is not tracked — see `assets/.gitignore` and `build.rs` — so there is nothing to
//! commit afterwards.

use ailoy_desktop_core::catalog::fetch_models_dev;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let data = fetch_models_dev().await?;
    let out = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/models.json");
    std::fs::create_dir_all(out.parent().unwrap())?;
    std::fs::write(&out, serde_json::to_vec_pretty(&data)?)?;
    let n: usize = data.providers.values().map(|p| p.models.len()).sum();
    let at = data
        .fetched_at
        .and_then(chrono::DateTime::from_timestamp_millis)
        .map(|t| t.to_rfc3339())
        .unwrap_or_default();
    println!(
        "wrote {} ({} providers, {n} models, fetched {at})",
        out.display(),
        data.providers.len()
    );
    Ok(())
}
