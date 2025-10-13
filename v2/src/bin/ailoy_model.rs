#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    ailoy::cli::ailoy_model::ailoy_model_cli(args).await
}
