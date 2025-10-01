use std::path::PathBuf;

use ailoy::cache::{Manifest, ManifestDirectory};
use anyhow::anyhow;
use aws_config::meta::region::ProvideRegion;
use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(version, about = "Ailoy Model Manager CLI", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    Upload {
        model_path: PathBuf,
        aws_profile_name: String,

        #[arg(short, long, default_value = None)]
        aws_endpoint_url: Option<String>,
    },
    Download {
        model_name: String,
        arch: String,
        os: String,
        device: String,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match &cli.command {
        Commands::Upload {
            model_path,
            aws_profile_name,
            aws_endpoint_url,
        } => {
            if !model_path.is_dir() {
                log::error!(
                    "Model path does not exist: {}",
                    model_path.to_string_lossy()
                );
                std::process::exit(1);
            }

            let bucket_name = "ailoy-cache";
            let model_name = model_path.file_name().unwrap().to_str().unwrap();
            let client = get_s3_client(aws_profile_name, aws_endpoint_url).await?;

            println!("Uploading model files from {:?}", model_path);

            let mut manifests = ManifestDirectory::new();
            for entry in std::fs::read_dir(model_path).unwrap() {
                let entry = entry.unwrap();
                let path = entry.path();

                if path.is_file() {
                    let filename = path.file_name().unwrap().to_str().unwrap();
                    if filename == "_manifest.json" {
                        continue;
                    }

                    println!(" - {:?}", filename);

                    let content = std::fs::read(path.clone())?;
                    let s3_body: aws_sdk_s3::primitives::ByteStream = content.into();
                    let manifest = Manifest::from_u8(s3_body.bytes().unwrap());
                    manifests.insert_file(
                        path.file_name().unwrap().to_str().unwrap().to_owned(),
                        manifest.clone(),
                    );

                    client
                        .put_object()
                        .bucket(bucket_name)
                        .key(format!("{}/{}", model_name, manifest.sha1()))
                        .body(s3_body)
                        .send()
                        .await
                        .map_err(|e| anyhow!(e))?;
                }
            }

            let manifests_json = serde_json::to_string(&manifests).unwrap().into_bytes();
            let body: aws_sdk_s3::primitives::ByteStream = manifests_json.into();
            client
                .put_object()
                .bucket(bucket_name)
                .key(format!("{}/_manifest.json", model_name))
                .body(body)
                .content_type("application/json")
                .send()
                .await
                .map_err(|e| anyhow!(e))?;
        }
        Commands::Download {
            model_name,
            arch,
            os,
            device,
        } => {
            println!("Download!");
        }
    }

    Ok(())
}

async fn get_s3_client(
    profile_name: &String,
    endpoint_url: &Option<String>,
) -> anyhow::Result<aws_sdk_s3::Client> {
    let region_provider: aws_config::profile::ProfileFileRegionProvider =
        aws_config::profile::ProfileFileRegionProvider::builder()
            .profile_name(profile_name)
            .build();
    let region = region_provider
        .region()
        .await
        .unwrap_or(aws_config::Region::new("auto"));

    let credentials_provider = aws_config::profile::ProfileFileCredentialsProvider::builder()
        .profile_name(profile_name)
        .build();

    let mut config_builder = aws_config::SdkConfig::builder()
        .behavior_version(aws_config::BehaviorVersion::latest())
        .credentials_provider(aws_sdk_s3::config::SharedCredentialsProvider::new(
            credentials_provider,
        ))
        .region(region);
    if let Some(endpoint_url) = endpoint_url {
        config_builder = config_builder.endpoint_url(endpoint_url);
    }
    let config = config_builder.build();

    let client = aws_sdk_s3::Client::new(&config);
    Ok(client)
}
