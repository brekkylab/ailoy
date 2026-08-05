//! ailoy guest init: mount the sandbox's virtio-fs/block devices with the
//! classic `mount(2)` syscall, then exec the payload.
//!
//! libkrun's `/init.krun` sets up the root, `/proc`-less, and hands control to a
//! configured exec path. That exec path is this binary. It exists because a guest
//! image's own `mount` binary cannot be relied on: util-linux (debian, most glibc
//! images) drives mounts through the new `fsopen`/`fsmount` API, which the libkrun
//! guest kernel rejects — while `mount(2)` itself works fine as root. busybox
//! (Alpine) calls `mount(2)` directly, which is the only reason the shell path
//! worked there. Calling `mount(2)` ourselves makes every image behave like
//! busybox did.
//!
//! Config comes in through the environment (small, fixed) so the large command
//! payload can still ride the ctrl virtio-fs share:
//!   AILOY_INIT_CTRL = "tag:/mountpoint"          (required)
//!   AILOY_INIT_WS   = "tag:/guest_root"          (optional workspace share)
//!   AILOY_INIT_DATA = "/dev/vda:/data"           (optional persistent upper)
//! Everything after `argv[0]` is the command to exec once mounts are in place.

use std::ffi::CString;
use std::process::exit;
use std::ptr;

/// `mount(2)`, creating the target directory first. Errors are returned, not
/// panicked, so a best-effort mount can be ignored by the caller.
fn mount(source: &str, target: &str, fstype: &str, data: Option<&str>) -> Result<(), i32> {
    let _ = std::fs::create_dir_all(target);
    let source = CString::new(source).unwrap();
    let target = CString::new(target).unwrap();
    let fstype = CString::new(fstype).unwrap();
    let data = data.map(|d| CString::new(d).unwrap());
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
    if rc == 0 {
        Ok(())
    } else {
        Err(unsafe { *libc::__errno_location() })
    }
}

/// Parse `tag:/mountpoint` (splitting on the first `:` only, so the mountpoint may
/// contain none).
fn spec(var: &str) -> Option<(String, String)> {
    let v = std::env::var(var).ok()?;
    let (a, b) = v.split_once(':')?;
    Some((a.to_string(), b.to_string()))
}

fn main() {
    // Pseudo-filesystems: best-effort. `/proc` in particular is what util-linux
    // and many runtimes expect; the guest kernel provides the rest of `/dev`.
    let _ = mount("proc", "/proc", "proc", None);
    let _ = mount("sysfs", "/sys", "sysfs", None);

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

    // The persistent upper (a block device). Best-effort and un-formatted here —
    // try the filesystems a freshly-provisioned image is formatted with; if none
    // mounts (e.g. a zeroed upper on first boot), leave `/data` absent rather than
    // fail the whole run.
    if let Some((dev, mp)) = spec("AILOY_INIT_DATA") {
        if mount(&dev, &mp, "ext4", None).is_err() {
            let _ = mount(&dev, &mp, "vfat", None);
        }
    }

    // Exec the payload (argv[1..]). execv replaces this process, so a return is
    // always an error.
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("ailoy-guest-init: no command to exec");
        exit(2);
    }
    let prog = CString::new(args[0].as_str()).unwrap();
    let cargs: Vec<CString> = args
        .iter()
        .map(|a| CString::new(a.as_str()).unwrap())
        .collect();
    let mut argv: Vec<*const libc::c_char> = cargs.iter().map(|c| c.as_ptr()).collect();
    argv.push(ptr::null());
    unsafe {
        libc::execv(prog.as_ptr(), argv.as_ptr());
    }
    let e = unsafe { *libc::__errno_location() };
    eprintln!("ailoy-guest-init: execv {} failed: errno {e}", args[0]);
    exit(127);
}
