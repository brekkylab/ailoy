use std::path::Path;

/// File logging under the app data dir; `RUST_LOG` filters, default `info`.
pub fn init(data_dir: &Path) {
    use tracing_subscriber::{EnvFilter, fmt, layer::SubscriberExt, util::SubscriberInitExt};
    let logs = data_dir.join("logs");
    let _ = std::fs::create_dir_all(&logs);
    let file = tracing_appender::rolling::daily(logs, "ailoy.log");
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    // `fs_timing` is a target of its own rather than a module path, so a `RUST_LOG` that
    // names a module — `ailoy_desktop_core=debug`, the one in the README — disables it
    // along with everything else it did not name, and a measurement run quietly measures
    // nothing. Put it back unless the variable mentions it, which is how it is turned off
    // on purpose (`RUST_LOG=fs_timing=off`).
    let filter = match std::env::var("RUST_LOG") {
        Ok(v) if v.contains("fs_timing") => filter,
        _ => filter.add_directive(
            "fs_timing=info"
                .parse()
                .expect("a literal directive parses"),
        ),
    };
    let _ = tracing_subscriber::registry()
        .with(filter)
        .with(fmt::layer().with_writer(file).with_ansi(false))
        .with(fmt::layer().with_writer(std::io::stderr))
        .try_init();
}
