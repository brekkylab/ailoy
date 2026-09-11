//! Regenerate `assets/models.json` from models.dev. Run from the repository root:
//! `cargo run -p ailoy-desktop-core --bin gen-catalog`.

use ailoy_desktop_core::catalog::{MODELS_DEV_URL, filter_models_dev};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let full: serde_json::Value = reqwest::get(MODELS_DEV_URL)
        .await?
        .error_for_status()?
        .json()
        .await?;
    let data = filter_models_dev(&full);
    let out = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/models.json");
    std::fs::create_dir_all(out.parent().unwrap())?;
    std::fs::write(&out, serde_json::to_vec_pretty(&data)?)?;
    let n: usize = data.providers.values().map(|p| p.models.len()).sum();
    println!(
        "wrote {} ({} providers, {n} models)",
        out.display(),
        data.providers.len()
    );
    Ok(())
}
