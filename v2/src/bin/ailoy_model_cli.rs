use std::path::PathBuf;

use ailoy::{
    cache::{Cache, Manifest, ManifestDirectory},
    model::get_accelerator,
};
use aws_config::meta::region::ProvideRegion;
use clap::{Parser, Subcommand};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use tokio::io::AsyncWriteExt;

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

        #[arg(short, long, default_value = "default")]
        aws_profile_name: String,

        #[arg(short, long, default_value = None)]
        aws_endpoint_url: Option<String>,

        #[arg(short, long, default_value = "ailoy-cache")]
        s3_bucket_name: String,
    },
    Download {
        model_name: String,

        #[arg(
            short,
            long,
            help = "Target platform triple (e.g. aarch64-apple-darwin). If not provided, it's inferred based on the current environment."
        )]
        platform: Option<String>,

        #[arg(
            short,
            long,
            help = "Target device (e.g. metal). If not provided, it's inferred based on the current environment."
        )]
        device: Option<String>,

        #[arg(short, long)]
        download_path: Option<PathBuf>,

        #[arg(short, long, default_value = "default")]
        aws_profile_name: String,

        #[arg(short, long)]
        aws_endpoint_url: Option<String>,

        #[arg(short, long, default_value = "ailoy-cache")]
        s3_bucket_name: String,
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

            let client = get_s3_client(aws_profile_name, aws_endpoint_url).await?;
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
                    .bucket(s3_bucket_name.clone())
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
            aws_profile_name,
            aws_endpoint_url,
            s3_bucket_name,
        } => {
            let model_name = model_name.clone().replace("/", "--");
            let platform = platform
                .clone()
                .unwrap_or(env!("BUILD_TARGET_TRIPLE").to_string());
            let device = device.clone().unwrap_or(get_accelerator().to_string());
            let download_path = download_path
                .clone()
                .unwrap_or(Cache::new().root().to_path_buf());

            println!("Downloading {}", model_name);
            println!("- Platform: {}", platform);
            println!("- Device: {}", device);
            println!("- Download path: {:?}", download_path);

            let client = get_s3_client(aws_profile_name, aws_endpoint_url).await?;

            let model_dirs = vec![
                model_name.clone(),
                format!("{}--{}--{}", model_name, platform, device),
            ];

            let mp = MultiProgress::new();
            let spinner_style = ProgressStyle::default_spinner()
                .template("{spinner:.green} {msg}")
                .unwrap()
                .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]);

            for model_dir in model_dirs.into_iter() {
                let dir_resp = client
                    .list_objects_v2()
                    .bucket(s3_bucket_name)
                    .prefix(format!("{}/", model_dir))
                    .max_keys(1)
                    .send()
                    .await?;
                let objects = dir_resp.contents();
                if objects.is_empty() {
                    eprintln!("Model directory '{}' not found in S3 bucket.", model_dir);
                    std::process::exit(1);
                }

                let manifest_obj = client
                    .get_object()
                    .bucket(s3_bucket_name)
                    .key(format!("{}/_manifest.json", model_dir))
                    .send()
                    .await
                    .map_err(|e| {
                        eprintln!(
                            "Failed to read _manifest.json for model directory '{}': {}",
                            model_dir,
                            e.to_string()
                        );
                        std::process::exit(1);
                    })?;
                let manifest_data = manifest_obj.body.collect().await?.into_bytes();
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

                    let pb = mp.add(ProgressBar::new_spinner());
                    pb.set_style(spinner_style.clone());
                    pb.set_message(local_key.clone());
                    pb.enable_steady_tick(std::time::Duration::from_millis(80));

                    let file_obj = client
                        .get_object()
                        .bucket(s3_bucket_name)
                        .key(remote_key)
                        .send()
                        .await?;
                    let file_data = file_obj.body.collect().await?.into_bytes();
                    let mut file =
                        tokio::fs::File::create(format!("{}/{}", download_dir, filename)).await?;
                    file.write_all(&file_data).await?;

                    pb.set_style(
                        ProgressStyle::default_spinner()
                            .template("{prefix:.green} {msg}")
                            .unwrap(),
                    );
                    pb.set_prefix("✓");
                    pb.finish();
                }
            }

            println!("🎉 Download complete!");
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
