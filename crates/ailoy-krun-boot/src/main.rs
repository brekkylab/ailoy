//! Host-side VM boot helper. `ailoy`'s `Sandbox::exec` materializes + ad-hoc-signs
//! the embedded copy of this binary and invokes it as a child, passing everything
//! it needs through the environment. `msb_krun::enter()` never returns (it
//! `_exit`s on guest shutdown), so this runs as a throwaway child, not in-process.
//!
//! The env contract and the serialized types below MUST stay in lockstep with
//! `ailoy`'s `src/runenv/sandbox.rs` (the parent writes these; here we read them).
//! It is the same cross-process contract the guest init already uses.

use base64::Engine as _;
use cortex::{PosixFs, VolumeSpec, Workspace, WorkspaceSpec};
use msb_krun::{DiskImageFormat, VmBuilder};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

const WORKSPACE_ENV: &str = "AILOY_KRUN_WORKSPACE";
/// Guest path the init binary is placed and exec'd at.
const GUEST_INIT_PATH: &str = "/.ailoy-init";
const DEFAULT_VCPUS: u8 = 2;
const DEFAULT_MEMORY_MIB: u32 = 2048;

/// The cortex workspace mounted into the sandbox, carried opaquely from the
/// parent. Mirrors `ailoy`'s `WorkspaceWire` — same fields, same serde.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct WorkspaceWire {
    tag: String,
    guest_root: String,
    spec: WorkspaceSpec,
}

/// Guest outbound-reach posture. Mirrors `ailoy`'s `SandboxNetwork`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum SandboxNetwork {
    Disabled,
    #[default]
    HostOnly,
    Public,
    Full,
}

/// Mirrors `ailoy`'s `NetworkPosture`.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct NetworkPosture {
    #[serde(default)]
    reach: SandboxNetwork,
    #[serde(default)]
    host_ports: Vec<u16>,
    #[serde(default)]
    domain_suffixes: Vec<String>,
}

impl NetworkPosture {
    /// Realize this posture as a smoltcp egress policy. Never called for
    /// `Disabled` (which attaches no device at all).
    fn policy(&self) -> microsandbox_network::policy::NetworkPolicy {
        use microsandbox_network::policy::{
            Action, Destination, DestinationGroup, Direction, DomainName, NetworkPolicy, PortRange,
            Protocol, Rule,
        };

        if matches!(self.reach, SandboxNetwork::Full) {
            return NetworkPolicy::allow_all();
        }

        // `Rule::allow_dns()` is narrow (UDP/TCP :53 to the gateway) and MUST come
        // first: under deny-by-default a policy without it refuses every DNS query.
        let mut rules = vec![Rule::allow_dns()];
        if matches!(self.reach, SandboxNetwork::Public) {
            rules.push(Rule::allow_egress(Destination::Group(
                DestinationGroup::Public,
            )));
        }
        rules.extend(self.host_ports.iter().map(|&port| Rule {
            direction: Direction::Egress,
            destination: Destination::Group(DestinationGroup::Host),
            protocols: vec![Protocol::Tcp],
            ports: vec![PortRange::single(port)],
            action: Action::Allow,
        }));
        rules.extend(self.domain_suffixes.iter().filter_map(|suffix| {
            match suffix.parse::<DomainName>() {
                Ok(name) => Some(Rule::allow_egress(Destination::DomainSuffix(name))),
                Err(e) => {
                    eprintln!("ailoy krun: ignoring invalid domain suffix '{suffix}': {e:?}");
                    None
                }
            }
        }));

        NetworkPolicy {
            default_egress: Action::Deny,
            default_ingress: Action::Deny,
            rules,
        }
    }
}

fn main() {
    let get = |k: &str| std::env::var(k).unwrap_or_else(|_| panic!("missing {k}"));
    let kernel = PathBuf::from(get("AILOY_KRUN_KERNEL"));
    let rootfs_erofs = PathBuf::from(get("AILOY_KRUN_ROOTFS_EROFS"));
    let upper = PathBuf::from(get("AILOY_KRUN_UPPER"));
    let console = PathBuf::from(get("AILOY_KRUN_CONSOLE"));
    let b64 = get("AILOY_KRUN_B64");
    // The parent (`Sandbox`) creates and owns a unique boot root per sandbox and
    // always passes its path — this helper never runs outside `Sandbox::exec`.
    let init_root = PathBuf::from(get("AILOY_KRUN_INIT_ROOT"));
    let workspace: Option<WorkspaceWire> = serde_json::from_str(&get(WORKSPACE_ENV))
        .unwrap_or_else(|e| panic!("parse {WORKSPACE_ENV}: {e}"));

    // The payload can be large (a `write` embedding base64 file bytes), so it
    // CANNOT ride in the exec argv (msb_krun puts args in the kernel cmdline, a
    // hard size limit). Decode it to a host file and carry it in over a dedicated
    // virtio-fs control mount; the exec argv stays a tiny fixed bootstrap.
    let payload = base64::engine::general_purpose::STANDARD
        .decode(&b64)
        .unwrap_or_else(|e| panic!("decode AILOY_KRUN_B64: {e}"));
    let ctrl_dir = std::env::temp_dir().join(format!("ailoy-ctrl-{}", std::process::id()));
    std::fs::create_dir_all(&ctrl_dir).unwrap_or_else(|e| panic!("create ctrl dir: {e}"));
    std::fs::write(ctrl_dir.join("run.sh"), &payload)
        .unwrap_or_else(|e| panic!("write ctrl run.sh: {e}"));
    let ctrl_backend = VolumeSpec::Local { host: ctrl_dir }
        .build()
        .unwrap_or_else(|e| panic!("build ctrl volume: {e}"));

    let bootstrap = "sh /.ailoyctrl/run.sh";

    let vcpus: u8 = std::env::var("AILOY_KRUN_VCPUS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_VCPUS);
    let memory_mib: u32 = std::env::var("AILOY_KRUN_MEMORY_MIB")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_MEMORY_MIB);

    let mut builder = VmBuilder::new()
        .machine(|m| m.vcpus(vcpus).memory_mib(memory_mib as usize))
        .kernel(|k| k.krunfw_path(&kernel))
        .fs(|fs| fs.root(&init_root))
        .fs(move |fs| fs.tag("ailoyctrl").custom(ctrl_backend))
        // vda: the writable ext4 upper. vdb: the read-only base image (EROFS).
        // Attach order fixes the guest device names.
        .disk(|d| d.path(&upper).format(DiskImageFormat::Raw))
        .disk(|d| {
            d.path(&rootfs_erofs)
                .read_only(true)
                .format(DiskImageFormat::Raw)
        })
        .console(|c| c.output(&console));

    // The mount spec the guest init needs for the workspace share (tag:guest_root),
    // captured before `workspace` is consumed below.
    let init_ws_env = workspace
        .as_ref()
        .map(|w| format!("{}:{}", w.tag, w.guest_root));

    // Realize the whole cortex workspace as one virtio-fs device: `from_spec`
    // rebuilds the unified tree, `PosixFs` binds it to msb_krun's `DynFileSystem`.
    if let Some(w) = workspace {
        let ws = Workspace::from_spec(&w.spec).unwrap_or_else(|e| {
            eprintln!("ailoy krun: build workspace: {e}");
            std::process::exit(1);
        });
        let backend: Box<dyn msb_krun::DynFileSystem + Send + Sync> = Box::new(PosixFs::new(ws));
        let tag = w.tag.clone();
        builder = builder.fs(move |fs| fs.tag(&tag).custom(backend));
    }

    // Guest networking. An in-process smoltcp userspace stack NATs guest egress to
    // the host under the posture's policy; its tokio runtime must outlive the VM.
    // `enter()` below diverges (`_exit`s on guest shutdown), so these guards live
    // for the VM's lifetime.
    let posture: NetworkPosture =
        serde_json::from_str(&std::env::var("AILOY_KRUN_NETWORK").unwrap_or_default())
            .unwrap_or_default();
    let mut net_prelude = String::new();
    let _net_guard: Option<(tokio::runtime::Runtime, microsandbox_network::network::SmoltcpNetwork)>;
    if posture.reach == SandboxNetwork::Disabled {
        _net_guard = None;
    } else {
        let _ = rustls::crypto::ring::default_provider().install_default();
        let mut netcfg = microsandbox_network::config::NetworkConfig::default();
        netcfg.policy = posture.policy();
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap_or_else(|e| panic!("ailoy krun: net runtime: {e}"));
        let mut stack = microsandbox_network::network::SmoltcpNetwork::new(netcfg, 0);
        stack.start(rt.handle().clone());
        let guest_mac = stack.guest_mac();
        let backend = stack.take_backend();
        net_prelude = guest_net_setup(&stack.guest_env_vars());
        builder = builder.net(move |n| n.mac(guest_mac).custom(backend));
        _net_guard = Some((rt, stack));
    }

    // The guest configures eth0 (if networking is on) before running the payload.
    let script = format!("{net_prelude}{bootstrap}");

    // Exec the guest init, not the shell directly: it overlays the ext4 upper
    // (/dev/vda) over the read-only image root, mounts the ctrl/workspace shares
    // via mount(2), then hands off to `/bin/sh -c <script>`. Mount targets travel
    // in the environment; the large payload stays on the ctrl share.
    let result = builder
        .exec(|e| {
            let mut e = e
                .path(GUEST_INIT_PATH)
                .env("AILOY_INIT_LOWER", "/dev/vdb")
                .env("AILOY_INIT_UPPER", "/dev/vda")
                .env("AILOY_INIT_CTRL", "ailoyctrl:/.ailoyctrl");
            if let Some(ws) = &init_ws_env {
                e = e.env("AILOY_INIT_WS", ws);
            }
            e.args(["/bin/sh", "-c", script.as_str()])
        })
        .build()
        .and_then(|vm| vm.enter());
    match result {
        Ok(never) => match never {},
        Err(e) => {
            eprintln!("ailoy krun boot: {e}");
            std::process::exit(1);
        }
    }
}

/// Emit the guest shell commands that bring up `eth0` with the static IPv4
/// address/gateway/DNS the smoltcp stack assigned (carried in its guest env vars
/// as `addr=<ip>/30,gw=<gw>,dns=<gw>`). The base rootfs has no network agent.
fn guest_net_setup(env_vars: &[(String, String)]) -> String {
    for (_key, val) in env_vars {
        if !(val.contains("addr=") && val.contains("gw=")) {
            continue;
        }
        let mut addr = None;
        let mut gw = None;
        for part in val.split(',') {
            if let Some(a) = part.strip_prefix("addr=") {
                addr = Some(a);
            } else if let Some(g) = part.strip_prefix("gw=") {
                gw = Some(g);
            }
        }
        if let (Some(addr), Some(gw)) = (addr, gw) {
            // Must be a single line: spliced into the exec argv, which msb_krun
            // writes to the kernel cmdline (newlines -> `InvalidAscii`).
            return format!(
                "ip link set eth0 up 2>/dev/null; \
                 ip addr add {addr} dev eth0 2>/dev/null; \
                 ip route add default via {gw} dev eth0 2>/dev/null; \
                 printf 'nameserver {gw}\\n' > /etc/resolv.conf 2>/dev/null; "
            );
        }
    }
    String::new()
}
