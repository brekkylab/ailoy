//! ailoy guest init: give the guest a writable, persistent root by overlaying an
//! ext4 upper (a per-sandbox block device) over the read-only image rootfs, then
//! mount the sandbox's virtio-fs shares and exec the payload.
//!
//! libkrun's `/init.krun` mounts the image rootfs (virtio-fs) at `/` and hands
//! control to a configured exec path — this binary. It exists because a guest
//! image's own `mount` binary cannot be relied on (util-linux, on most glibc
//! images, drives mounts through the new `fsopen`/`fsmount` API the libkrun
//! kernel rejects; `mount(2)` itself works fine), so every mount here is a direct
//! `mount(2)` syscall — the same approach microsandbox's `agentd` took.
//!
//! With `AILOY_INIT_UPPER` set, it builds an overlay root so writes anywhere in
//! the guest land on the upper (and thus persist across the ephemeral per-exec
//! VMs) while the image stays pristine. Config comes in through the environment;
//! the large command payload rides the ctrl virtio-fs share:
//!   AILOY_INIT_UPPER = "/dev/vda"                (overlay upper block device)
//!   AILOY_INIT_CTRL  = "tag:/mountpoint"         (required)
//!   AILOY_INIT_WS    = "tag:/guest_root"         (optional workspace share)
//! Everything after argv[0] is the command to exec once the root is ready.

use std::ffi::CString;
use std::process::exit;
use std::ptr;

fn cstr(s: &str) -> CString {
    CString::new(s).unwrap()
}

fn errno() -> i32 {
    unsafe { *libc::__errno_location() }
}

/// `mount(2)`, creating the target directory first. Errors are returned as errno,
/// not panicked, so a best-effort mount can be ignored by the caller.
fn mount(source: &str, target: &str, fstype: &str, data: Option<&str>) -> Result<(), i32> {
    let _ = std::fs::create_dir_all(target);
    let (source, target, fstype) = (cstr(source), cstr(target), cstr(fstype));
    let data = data.map(cstr);
    let data_ptr = data
        .as_ref()
        .map(|c| c.as_ptr() as *const libc::c_void)
        .unwrap_or(ptr::null());
    let rc = unsafe {
        libc::mount(
            source.as_ptr(),
            target.as_ptr(),
            fstype.as_ptr(),
            0,
            data_ptr,
        )
    };
    if rc == 0 { Ok(()) } else { Err(errno()) }
}

/// Mount the standard pseudo-filesystems (best-effort — they may already be up,
/// or be re-established in a fresh root after a pivot).
fn mount_pseudo() {
    let _ = mount("proc", "/proc", "proc", None);
    let _ = mount("devtmpfs", "/dev", "devtmpfs", None);
    let _ = mount("sysfs", "/sys", "sysfs", None);
}

/// Parse `tag:/mountpoint` (splitting on the first `:` only).
fn spec(var: &str) -> Option<(String, String)> {
    let v = std::env::var(var).ok()?;
    let (a, b) = v.split_once(':')?;
    Some((a.to_string(), b.to_string()))
}

/// Build an overlay root: the read-only image EROFS (`lower_dev`) as the lower,
/// the ext4 (`upper_dev`) as the writable upper, then `pivot_root` into it. After
/// this returns, `/` is the writable overlay backed by the image + persistent
/// upper — both block devices, so overlay copy-up works (a virtio-fs lower does
/// not).
fn setup_overlay_root(lower_dev: &str, upper_dev: &str) -> Result<(), String> {
    // pivot_root refuses (EINVAL) while the root mount has shared propagation —
    // make the whole tree private first.
    let rc = unsafe {
        libc::mount(
            ptr::null(),
            cstr("/").as_ptr(),
            ptr::null(),
            libc::MS_REC | libc::MS_PRIVATE,
            ptr::null(),
        )
    };
    if rc != 0 {
        return Err(format!("make-rprivate /: errno {}", errno()));
    }

    // A tmpfs scratch area so the plumbing dirs live off the (RO) image lower.
    mount("tmpfs", "/mnt", "tmpfs", None).map_err(|e| format!("tmpfs /mnt: errno {e}"))?;
    let _ = std::fs::create_dir_all("/mnt/lower");
    let _ = std::fs::create_dir_all("/mnt/upper");
    let _ = std::fs::create_dir_all("/mnt/newroot");

    // The lower disk is attached read-only, so the mount must pass MS_RDONLY or
    // the kernel refuses it with EACCES.
    {
        let (s, t, f) = (cstr(lower_dev), cstr("/mnt/lower"), cstr("erofs"));
        let rc =
            unsafe { libc::mount(s.as_ptr(), t.as_ptr(), f.as_ptr(), libc::MS_RDONLY, ptr::null()) };
        if rc != 0 {
            return Err(format!("mount {lower_dev} erofs: errno {}", errno()));
        }
    }
    mount(upper_dev, "/mnt/upper", "ext4", None)
        .map_err(|e| format!("mount {upper_dev} ext4: errno {e}"))?;
    let _ = std::fs::create_dir_all("/mnt/upper/upper");
    let _ = std::fs::create_dir_all("/mnt/upper/work");

    mount(
        "overlay",
        "/mnt/newroot",
        "overlay",
        Some("lowerdir=/mnt/lower,upperdir=/mnt/upper/upper,workdir=/mnt/upper/work"),
    )
    .map_err(|e| format!("mount overlay: errno {e}"))?;

    // pivot_root(new_root=".", put_old="oldroot") with cwd = new_root.
    let _ = std::fs::create_dir_all("/mnt/newroot/oldroot");
    if unsafe { libc::chdir(cstr("/mnt/newroot").as_ptr()) } != 0 {
        return Err(format!("chdir newroot: errno {}", errno()));
    }
    let rc = unsafe {
        libc::syscall(
            libc::SYS_pivot_root,
            cstr(".").as_ptr(),
            cstr("oldroot").as_ptr(),
        )
    };
    if rc != 0 {
        return Err(format!("pivot_root: errno {}", errno()));
    }
    if unsafe { libc::chroot(cstr(".").as_ptr()) } != 0 {
        return Err(format!("chroot: errno {}", errno()));
    }
    if unsafe { libc::chdir(cstr("/").as_ptr()) } != 0 {
        return Err(format!("chdir /: errno {}", errno()));
    }
    // Detach the old image root. Lazy: the overlay keeps its lower/upper/work
    // handles alive internally even once they leave the namespace.
    if unsafe { libc::umount2(cstr("/oldroot").as_ptr(), libc::MNT_DETACH) } != 0 {
        return Err(format!("umount oldroot: errno {}", errno()));
    }
    let _ = std::fs::remove_dir("/oldroot");
    Ok(())
}

fn main() {
    mount_pseudo();

    // Overlay the persistent upper over the image EROFS so writes persist across
    // execs while the image stays read-only.
    if let (Ok(lower), Ok(upper)) = (
        std::env::var("AILOY_INIT_LOWER"),
        std::env::var("AILOY_INIT_UPPER"),
    ) {
        if let Err(e) = setup_overlay_root(&lower, &upper) {
            eprintln!("ailoy-guest-init: {e}");
            exit(1);
        }
        // The overlay's /proc, /dev, /sys are the image's empty dirs — re-mount.
        mount_pseudo();
    }

    // The ctrl share carries the command payload — without it there is nothing to
    // run, so a failure here is worth reporting.
    if let Some((tag, mp)) = spec("AILOY_INIT_CTRL") {
        if let Err(e) = mount(&tag, &mp, "virtiofs", None) {
            eprintln!("ailoy-guest-init: mount ctrl {tag} -> {mp} failed: errno {e}");
        }
    }
    // The workspace share (cortex). Optional: a bare `exec` sandbox has none.
    if let Some((tag, mp)) = spec("AILOY_INIT_WS") {
        if let Err(e) = mount(&tag, &mp, "virtiofs", None) {
            eprintln!("ailoy-guest-init: mount workspace {tag} -> {mp} failed: errno {e}");
        }
    }

    // Exec the payload (argv[1..]). execv replaces this process, so a return is
    // always an error.
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("ailoy-guest-init: no command to exec");
        exit(2);
    }
    let prog = cstr(&args[0]);
    let cargs: Vec<CString> = args.iter().map(|a| cstr(a)).collect();
    let mut argv: Vec<*const libc::c_char> = cargs.iter().map(|c| c.as_ptr()).collect();
    argv.push(ptr::null());
    unsafe {
        libc::execv(prog.as_ptr(), argv.as_ptr());
    }
    eprintln!("ailoy-guest-init: execv {} failed: errno {}", args[0], errno());
    exit(127);
}
