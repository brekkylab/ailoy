use std::process::Command;

fn main() -> anyhow::Result<()> {
    #[cfg(feature = "python")]
    {
        // `stub_info` is a function defined by `define_stub_info_gatherer!` macro.
        let stub = ailoy::ffi::py::stub_info()?;
        stub.generate()?;

        // Inject "CacheResultT" typevar
        inject_typevar(&stub, "CacheResultT")?;
    }
    Ok(())
}

fn inject_typevar(stub: &pyo3_stub_gen::StubInfo, typevar: &str) -> anyhow::Result<()> {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")?;
    let script_path = std::path::Path::new(&manifest_dir)
        .join("src")
        .join("bin")
        .join("inject_typevar.py");
    let stub_filepath = stub.python_root.join("ailoy/_core.pyi");
    let output = Command::new("python3")
        .arg(&script_path)
        .arg(&stub_filepath)
        .arg("CacheResultT")
        .output()?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("Failed to inject TypeVar: {}", stderr);
    }
    println!(
        "Successfully injected TypeVar '{}' into {}",
        typevar,
        stub_filepath.display()
    );
    Ok(())
}
