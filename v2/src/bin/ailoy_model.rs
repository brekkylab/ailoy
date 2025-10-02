#[tokio::main]
async fn main() -> anyhow::Result<()> {
    ailoy::cli::ailoy_model::ailoy_model_cli().await
}
