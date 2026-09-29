//! Run OpenTTD in a console, and play it from here over VNC.
//!
//! ```sh
//! cargo run --example openttd
//! ```
//!
//! [OpenTTD](https://www.openttd.org) is the open-source game after Transport Tycoon Deluxe,
//! and with OpenGFX, OpenSFX and OpenMSX, everything it needs is free: there is nothing to
//! bring. The console installs it all from Debian, starts it in a window on a virtual
//! display, and shows that display with a VNC server. No agent: this is only the game,
//! running in a console.
//!
//! **Watching and playing**: nothing can connect into a console, so `tunnel.py` connects out
//! to this program, which joins it to a viewer on `localhost:5901`. Open
//! `vnc://localhost:5901` (Screen Sharing on macOS, or any VNC viewer) with the password
//! `openttd`. The mouse and the keyboard both work.
//!
//! * `artifacts/` at `/artifacts`, writable — OpenTTD's own folders, in `user/openttd/`: its
//!   settings, saved games and screenshots, kept from one run to the next; and the logs, in
//!   `logs/`.
//!
//! Environment:
//!
//! * `OPENTTD_SIZE` — the display, `1280x720` by default.
//! * `OPENTTD_VNC_PASSWORD` — the viewer's password, `openttd` by default.

use std::{collections::VecDeque, path::Path, sync::Arc, time::Duration};

use anyhow::{Context as _, bail};
use cortex::{console::ConsoleClient, image::Recipe, protocol::NetworkAccess};
use tokio::{
    io::{AsyncReadExt as _, AsyncWriteExt as _},
    net::{TcpListener, TcpStream},
    sync::Mutex,
};

/// The port on this machine the console's tunnel connects to: the one port it is granted.
const TUNNEL_PORT: u16 = 5500;

/// The port a VNC viewer connects to here.
const VIEWER_PORT: u16 = 5901;

/// What the console sees this machine as: its gateway, which opens a granted port on
/// loopback here.
const HOST_FROM_CONSOLE: &str = "10.0.2.1";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let size = std::env::var("OPENTTD_SIZE").unwrap_or_else(|_| "1280x720".to_string());
    let password = std::env::var("OPENTTD_VNC_PASSWORD").unwrap_or_else(|_| "openttd".to_string());

    // Absolute, because a mount is named to the server as a `file://` URL.
    let project_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/openttd");
    let artifacts = project_path.join("artifacts");
    std::fs::create_dir_all(&artifacts)
        .with_context(|| format!("creating {}", artifacts.display()))?;

    // Bound before the console starts, so the tunnel has somewhere to connect to at once.
    let from_console = TcpListener::bind(("127.0.0.1", TUNNEL_PORT))
        .await
        .with_context(|| format!("listening on {TUNNEL_PORT} for the console"))?;
    let viewers = TcpListener::bind(("127.0.0.1", VIEWER_PORT))
        .await
        .with_context(|| format!("listening on {VIEWER_PORT} for a viewer"))?;
    tokio::spawn(tunnel(from_console, viewers));

    let mut console = ConsoleClient::builder()
        .image(
            // All of it in `main`: the game, and the free graphics, sounds and music it needs.
            Recipe::new("debian:trixie-slim").step(
                "apt-get update \
                && apt-get install -y --no-install-recommends \
                    openttd openttd-opengfx openttd-opensfx openttd-openmsx xvfb x11vnc python3 \
                && rm -rf /var/lib/apt/lists/*",
            ),
        )
        // For `start.sh` and `tunnel.py`.
        .mount_readonly(project_path.clone(), "/example")
        .mount(artifacts, "/artifacts")
        // Nothing outside, and of this machine only the tunnel's port.
        .network(NetworkAccess::host().with_host_ports([TUNNEL_PORT]))
        .vcpus(2)
        .memory_mib(2048)
        .build()
        .await
        .with_context(|| "starting the console")?;

    let out = console
        .exec(
            [
                "sh",
                "/example/start.sh",
                &size,
                &format!("{HOST_FROM_CONSOLE}:{TUNNEL_PORT}"),
                &password,
            ],
            Some(60_000),
        )
        .await?;
    print!("{}", String::from_utf8_lossy(&out.stdout));
    if out.code != 0 {
        bail!(
            "starting OpenTTD:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    println!(
        "OpenTTD is running. Play it at vnc://localhost:{VIEWER_PORT} (password: {password}). \
        Press Enter here, or quit the game, to stop."
    );

    // Until Enter here, or until the game is gone.
    let enter = tokio::task::spawn_blocking(|| std::io::stdin().read_line(&mut String::new()));
    tokio::pin!(enter);
    loop {
        tokio::select! {
            _ = &mut enter => break,
            _ = tokio::time::sleep(Duration::from_secs(2)) => {
                let alive = console
                    .exec(["sh", "-c", "kill -0 \"$(cat /tmp/openttd.pid)\""], Some(10_000))
                    .await?;
                if alive.code != 0 {
                    println!("OpenTTD has stopped.");
                    break;
                }
            }
        }
    }

    // Dropping the console tears it down, game and all.
    Ok(())
}

/// Join viewers here to the VNC server in the console, which no one here can connect to.
///
/// The console's `tunnel.py` keeps one connection open to `from_console`, saying `CTRL`. For
/// each viewer that connects to `viewers`, this writes `OPEN` on that connection, the tunnel
/// connects again saying `DATA` with the server on its other end, and the two are joined.
async fn tunnel(from_console: TcpListener, viewers: TcpListener) {
    #[derive(Default)]
    struct Tunnel {
        ctrl: Option<TcpStream>,
        waiting: VecDeque<TcpStream>,
    }
    let tunnel = Arc::new(Mutex::new(Tunnel::default()));

    let shared = tunnel.clone();
    tokio::spawn(async move {
        while let Ok((viewer, _)) = viewers.accept().await {
            let mut t = shared.lock().await;
            t.waiting.push_back(viewer);
            // With no control connection yet, the viewer waits for one: it asks for them all.
            if let Some(ctrl) = t.ctrl.as_mut()
                && ctrl.write_all(b"OPEN\n").await.is_err()
            {
                t.ctrl = None;
            }
        }
    });

    while let Ok((mut conn, _)) = from_console.accept().await {
        let tunnel = tunnel.clone();
        tokio::spawn(async move {
            let mut hello = [0u8; 5];
            if conn.read_exact(&mut hello).await.is_err() {
                return;
            }
            let mut t = tunnel.lock().await;
            match &hello {
                b"CTRL\n" => {
                    for _ in 0..t.waiting.len() {
                        if conn.write_all(b"OPEN\n").await.is_err() {
                            return;
                        }
                    }
                    t.ctrl = Some(conn);
                }
                b"DATA\n" => {
                    if let Some(mut viewer) = t.waiting.pop_front() {
                        drop(t);
                        let _ = tokio::io::copy_bidirectional(&mut viewer, &mut conn).await;
                    }
                }
                _ => {}
            }
        });
    }
}
