use std::path::PathBuf;

use aws_config::meta::region::ProvideRegion;
use clap::{Parser, Subcommand};
use futures::stream::StreamExt;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use tokio::io::AsyncWriteExt;
use url::Url;

use crate::{
    cache::{Cache, Manifest, ManifestDirectory},
    model::get_accelerator,
};

#[derive(Parser, Debug)]
#[command(name = "ailoy-model", version, about = "Ailoy Model Manager CLI", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    Upload {
        model_path: PathBuf,

        #[arg(long, default_value = "default")]
        aws_profile_name: String,

        #[arg(long, default_value = None)]
        aws_endpoint_url: Option<String>,

        #[arg(long, default_value = "ailoy-cache")]
        s3_bucket_name: String,
    },
    Download {
        model_name: String,

        #[arg(
            long,
            help = "Target platform triple (e.g. aarch64-apple-darwin). If not provided, it's inferred based on the current environment."
        )]
        platform: Option<String>,

        #[arg(
            long,
            help = "Target device (e.g. metal). If not provided, it's inferred based on the current environment."
        )]
        device: Option<String>,

        #[arg(
            long,
            help = "Path to download model files. Default to Ailoy cache root (`$HOME/.cache/ailoy`)"
        )]
        download_path: Option<PathBuf>,

        #[arg(
            long,
            help = "Remote URL to download model files from. Default to Ailoy cache remote url"
        )]
        cache_remote_url: Option<Url>,
    },
}

pub async fn ailoy_model_cli(args: Vec<String>) -> anyhow::Result<()> {
    let cli = Cli::parse_from(args.clone());

    match &cli.command {
        Commands::Upload {
            model_path,
            aws_profile_name,
            aws_endpoint_url,
            s3_bucket_name,
        } => {
            if !model_path.is_dir() {
                eprintln!(
                    "Model path does not exist: {}",
                    model_path.to_string_lossy()
                );
                std::process::exit(1);
            }

            let model_name = model_path.file_name().unwrap().to_str().unwrap();

            let client = get_s3_client(&aws_profile_name, &aws_endpoint_url).await?;
            let target_entries = std::fs::read_dir(model_path)
                .unwrap()
                .into_iter()
                .map(|entry| entry.unwrap().path())
                .filter(|path| {
                    let filename = path.file_name().unwrap().to_str().unwrap();
                    path.is_file() && filename != "_manifest.json"
                })
                .collect::<Vec<_>>();
            let mut manifests = ManifestDirectory::new();

            let mp = MultiProgress::new();
            let spinner_style = ProgressStyle::default_spinner()
                .template("{spinner:.green} {msg}")
                .unwrap()
                .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]);

            println!("Uploading model files from {:?}", model_path);
            for path in target_entries {
                let filename = path.file_name().unwrap().to_string_lossy().to_string();

                let pb = mp.add(ProgressBar::new_spinner());
                pb.set_style(spinner_style.clone());
                pb.set_message(filename.clone());
                pb.enable_steady_tick(std::time::Duration::from_millis(80));

                let content = std::fs::read(path.clone())?;
                let s3_body: aws_sdk_s3::primitives::ByteStream = content.into();
                let manifest = Manifest::from_u8(s3_body.bytes().unwrap());
                manifests.insert_file(
                    path.file_name().unwrap().to_str().unwrap().to_owned(),
                    manifest.clone(),
                );

                match client
                    .put_object()
                    .bucket(s3_bucket_name)
                    .key(format!("{}/{}", model_name, manifest.sha1()))
                    .body(s3_body)
                    .send()
                    .await
                {
                    Ok(_) => {
                        pb.set_style(
                            ProgressStyle::default_spinner()
                                .template("{prefix:.green} {msg}")
                                .unwrap(),
                        );
                        pb.set_prefix("✓");
                        pb.finish();
                    }
                    Err(e) => {
                        pb.set_style(
                            ProgressStyle::default_spinner()
                                .template("{prefix:.red} {msg}")
                                .unwrap(),
                        );
                        pb.set_prefix("x");
                        pb.finish();
                        eprintln!("Failed to upload {}: {}", filename, e);
                        std::process::exit(1);
                    }
                }
            }

            let manifests_json = serde_json::to_string(&manifests).unwrap().into_bytes();
            let body: aws_sdk_s3::primitives::ByteStream = manifests_json.into();
            client
                .put_object()
                .bucket(s3_bucket_name)
                .key(format!("{}/_manifest.json", model_name))
                .body(body)
                .content_type("application/json")
                .send()
                .await
                .map_err(|e| {
                    eprintln!("Failed to upload _manifest.json: {}", e);
                    std::process::exit(1);
                })?;

            println!("🎉 Upload complete!")
        }
        Commands::Download {
            model_name,
            platform,
            device,
            download_path,
            cache_remote_url,
        } => {
            let model_name = model_name.clone().replace("/", "--");
            let platform = platform
                .clone()
                .unwrap_or(env!("BUILD_TARGET_TRIPLE").to_string());
            let device = device.clone().unwrap_or(get_accelerator().to_string());

            let cache = Cache::new();
            let cache_remote_url = cache_remote_url
                .clone()
                .unwrap_or(cache.remote_url().clone());
            let download_path = download_path.clone().unwrap_or(cache.root().to_path_buf());

            println!("Downloading {}", model_name);
            println!("- Platform: {}", platform);
            println!("- Device: {}", device);
            println!("- Download path: {:?}", download_path);

            let model_dirs = vec![
                model_name.clone(), // common files (e.g. model params, tokenizer config, etc.)
                format!("{}--{}--{}", model_name, platform, device), // platform and device specific files (e.g. runtime lib)
            ];

            let pb_style = ProgressStyle::default_bar().template("{spinner:.green} {msg} [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} [{elapsed_precise}]")?.progress_chars("#>-");

            let client = reqwest::Client::new();

            for model_dir in model_dirs.into_iter() {
                let manifest_url =
                    cache_remote_url.join(format!("{}/_manifest.json", model_dir).as_str())?;
                let manifest_resp = client.get(manifest_url).send().await?;
                let manifest_data = manifest_resp.bytes().await?.to_vec();
                let manifest_dir: ManifestDirectory =
                    serde_json::from_slice(manifest_data.iter().as_slice())?;

                let download_dir = format!("{}/{}", download_path.to_str().unwrap(), model_dir);
                std::fs::create_dir_all(&download_dir).map_err(|e| {
                    eprintln!(
                        "Failed to create directory {:?}: {}",
                        download_dir,
                        e.to_string()
                    );
                    std::process::exit(1);
                })?;

                for (filename, manifest) in manifest_dir.files.iter() {
                    let remote_key = format!("{}/{}", model_dir, manifest.sha1());
                    let local_key = format!("{}/{}", model_dir, filename);

                    let file_url = cache_remote_url.join(&remote_key)?;
                    let file_resp = client.get(file_url).send().await?.error_for_status()?;
                    let total_size = file_resp.content_length().unwrap_or(0);

                    let pb = ProgressBar::new(total_size);
                    pb.set_style(pb_style.clone());
                    pb.set_message(local_key.clone());

                    let mut file =
                        tokio::fs::File::create(format!("{}/{}", download_dir, filename)).await?;
                    let mut stream = file_resp.bytes_stream();
                    let mut downloaded: u64 = 0;

                    while let Some(chunk_result) = stream.next().await {
                        let chunk = chunk_result?;
                        file.write_all(&chunk).await?;
                        let new = u64::min(downloaded + (chunk.len() as u64), total_size);
                        downloaded = new;
                        pb.set_position(new);
                    }

                    pb.finish();
                }
            }

            println!("\n🎉 Download complete!");
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
