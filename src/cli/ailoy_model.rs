use std::{io::Write as _, path::PathBuf};

use aws_config::meta::region::ProvideRegion;
use clap::{Parser, Subcommand};
use futures::stream::StreamExt;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use tokio::io::AsyncWriteExt;
use url::Url;

use crate::{
    cache::{Cache, Manifest, ManifestDirectory},
    constants::AILOY_VERSION,
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
    List,
    Remove {
        model_name: String,
    },
    Upload {
        model_path: PathBuf,

        #[arg(
            long,
            help = "Dump files and manifest to local directory instead of uploading to remote S3 bucket. Use this as debugging or DRY option."
        )]
        to_local_path: Option<PathBuf>,

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

        #[arg(
            long,
            help = format!("Target Ailoy version. Default: {}", AILOY_VERSION)
        )]
        ailoy_version: Option<String>,
    },
}

pub async fn ailoy_model_cli(args: Vec<String>) -> anyhow::Result<()> {
    let cli = Cli::parse_from(args.clone());

    fn dir_size(path: &std::path::Path) -> u64 {
        let mut total = 0;
        if let Ok(entries) = std::fs::read_dir(path) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    total += dir_size(&path);
                } else if let Ok(metadata) = std::fs::metadata(&path) {
                    total += metadata.len();
                }
            }
        }
        total
    }

    match &cli.command {
        Commands::List => {
            let cache = Cache::new();
            let root = cache.root();
            println!("Cache root: {:?}", root);

            // Check if the cache root exists
            if !root.exists() {
                return Err(anyhow::anyhow!(
                    "Cache root directory does not exist: {:?}",
                    root
                ));
            }

            // Iterate through all entries in the cache root
            let entries = std::fs::read_dir(root).expect("Failed to read cache root directory");
            let mut models = std::collections::BTreeMap::<String, u64>::new();

            for entry in entries {
                if let Ok(entry) = entry {
                    let path = entry.path();

                    // Only consider directories
                    if path.is_dir() {
                        if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                            let parts: Vec<&str> = name.split("--").collect();

                            // Ignore entries with 0 or 1 parts
                            if parts.len() < 2 {
                                continue;
                            }

                            // Take first two elements and join with "/"
                            let parsed = format!("{}/{}", parts[0], parts[1]);
                            let size = dir_size(&path);
                            *models.entry(parsed).or_insert(0) += size;
                        }
                    }
                }
            }

            for (model_name, model_size) in &models {
                println!(
                    "* {} ({:.2} MB)",
                    model_name,
                    *model_size as f64 / 1024.0 / 1024.0
                );
            }
        }
        Commands::Remove { model_name } => {
            let cache = Cache::new();
            let root = cache.root();
            println!("Cache root: {:?}", root);

            // Check if the cache root exists
            if !root.exists() {
                return Err(anyhow::anyhow!(
                    "Cache root directory does not exist: {:?}",
                    root
                ));
            }

            let mut parts = model_name.splitn(3, '/');
            let kind = parts.next().unwrap_or("").trim();
            let name = parts.next().unwrap_or("").trim();

            if kind.is_empty() || name.is_empty() {
                return Err(anyhow::anyhow!(
                    "Invalid model name '{}': expected format 'org/model-name'",
                    model_name
                ));
            }

            let prefix = format!("{}--{}", kind, name);
            let entries = std::fs::read_dir(root)?;

            let mut targets = Vec::new();
            let mut total_size = 0u64;

            for entry in entries {
                let entry = entry?;
                let path = entry.path();

                if !path.is_dir() {
                    continue;
                }

                if let Some(dir_name) = path.file_name().and_then(|n| n.to_str()) {
                    if dir_name.starts_with(&prefix) {
                        let sz = dir_size(&path);
                        targets.push((path.clone(), dir_name.to_string(), sz));
                        total_size += sz;
                    }
                }
            }

            if targets.is_empty() {
                println!(
                    "No directories found for '{}'. Nothing to remove.",
                    model_name
                );
                return Ok(());
            }

            println!("-----------------------------------------------");
            for (_, name, sz) in &targets {
                println!("{:<40} {:>10.2} MB", name, *sz as f64 / 1024.0 / 1024.0);
            }
            println!(
                "About to remove {} directories, freeing approximately {:.2} MB total.",
                targets.len(),
                total_size as f64 / 1024.0 / 1024.0
            );

            // Confirmation prompt
            print!("Proceed with deletion? (y/N): ");
            std::io::stdout().flush().unwrap();

            let mut input = String::new();
            std::io::stdin().read_line(&mut input)?;
            let input = input.trim().to_lowercase();

            if input == "y" {
                let mut removed = Vec::new();
                let mut total_freed = 0u64;

                for (path, name, sz) in &targets {
                    std::fs::remove_dir_all(path)?;
                    removed.push((name.clone(), *sz));
                    total_freed += sz;
                }
                println!(
                    "Removed, freed {:.2} MB total.",
                    total_freed as f64 / 1024.0 / 1024.0
                );
            } else {
                println!("Aborted.");
            }
        }
        Commands::Upload {
            model_path,
            to_local_path,
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

            let s3_client: Option<aws_sdk_s3::Client> = if to_local_path.is_none() {
                Some(get_s3_client(&aws_profile_name, &aws_endpoint_url).await?)
            } else {
                None
            };

            println!("Uploading model files from {:?}", model_path);
            if let Some(local_path) = to_local_path {
                tokio::fs::create_dir_all(local_path.join(model_name)).await?;
                println!("- To local path: {:?}", local_path);
            }

            for path in target_entries {
                let filename = path.file_name().unwrap().to_string_lossy().to_string();

                let content = std::fs::read(path.clone())?;
                let manifest = Manifest::from_u8(&content);
                manifests.insert_file(
                    path.file_name().unwrap().to_str().unwrap().to_owned(),
                    manifest.clone(),
                );

                let pb = mp.add(ProgressBar::new_spinner());
                pb.set_style(spinner_style.clone());
                pb.set_message(format!("{} (sha1: {})", filename, manifest.sha1()));
                pb.enable_steady_tick(std::time::Duration::from_millis(80));

                let upload_result: anyhow::Result<()> = if let Some(client) = &s3_client {
                    let s3_body: aws_sdk_s3::primitives::ByteStream = content.into();
                    client
                        .put_object()
                        .bucket(s3_bucket_name)
                        .key(format!("{}/{}", model_name, manifest.sha1()))
                        .body(s3_body)
                        .send()
                        .await?;
                    Ok(())
                } else {
                    let local_path = to_local_path.as_ref().unwrap();
                    let mut file =
                        tokio::fs::File::create(local_path.join(model_name).join(manifest.sha1()))
                            .await?;
                    file.write_all(&content).await?;
                    Ok(())
                };

                match upload_result {
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
            if let Some(client) = &s3_client {
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
            } else {
                let local_path = to_local_path.as_ref().unwrap();
                let mut file =
                    tokio::fs::File::create(local_path.join(model_name).join("_manifest.json"))
                        .await?;
                file.write_all(&manifests_json).await?;
            }

            println!("\n🎉 Upload complete!")
        }
        Commands::Download {
            model_name,
            platform,
            device,
            download_path,
            cache_remote_url,
            ailoy_version,
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

            let ailoy_version = ailoy_version.clone().unwrap_or(AILOY_VERSION.to_string());

            println!("Downloading {}", model_name);
            println!("- Platform: {}", platform);
            println!("- Device: {}", device);
            println!("- Download path: {:?}", download_path);
            println!("- Target Ailoy version: {}", ailoy_version);

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
                let manifest_text = manifest_resp.text().await?;
                let manifest_dir = serde_json::from_str::<ManifestDirectory>(&manifest_text)?;

                let download_dir = format!("{}/{}", download_path.to_str().unwrap(), model_dir);
                std::fs::create_dir_all(&download_dir).map_err(|e| {
                    eprintln!(
                        "Failed to create directory {:?}: {}",
                        download_dir,
                        e.to_string()
                    );
                    std::process::exit(1);
                })?;

                for (filename, manifest) in manifest_dir.get_file_manifests(ailoy_version.as_str())
                {
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
