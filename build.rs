use std::{env, path::PathBuf};

fn main() {
    // Set target triple
    let target = std::env::var("TARGET").expect("TARGET not set");
    println!("cargo:rustc-env=BUILD_TARGET_TRIPLE={}", target);

    // Load .env file at build time if exists
    if let Ok(path) = std::env::var("CARGO_MANIFEST_DIR") {
        let env_path = std::path::Path::new(&path).join(".env");
        if env_path.exists() {
            for item in dotenvy::dotenv_iter().expect("Failed to read .env file") {
                let (key, value) = item.expect("Failed to parse .env line");
                println!("cargo:rustc-env={}={}", key, value);
            }
        }
    }

    if target.starts_with("wasm") {
        build_wasm();
        return;
    } else {
        build_native();
        return;
    }
}

fn build_native() {
    let tvm_ffi_lib = std::env::var("DEP_TVM_FFI_LIB").expect("tvm-ffi-sys did not export lib");
    let tvm_runtime_lib =
        std::env::var("DEP_TVM_RUNTIME_LIB").expect("tvm-runtime-sys did not export lib");

    #[cfg(target_family = "unix")]
    {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", tvm_ffi_lib);
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", tvm_runtime_lib);
    }
    #[cfg(target_family = "windows")]
    {
        println!("cargo:rustc-link-search=native={}", tvm_ffi_lib);
        println!("cargo:rustc-link-search=native={}", tvm_runtime_lib);
    }

    if std::env::var_os("CARGO_FEATURE_NODEJS").is_some() {
        napi_build::setup();
    }
}

fn build_wasm() {
    use std::process::Command;

    println!("cargo:rustc-cfg=feature=\"wasm\"");

    let cargo_manifest_dir =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set"));
    let shim_js_dir = cargo_manifest_dir.join("shim_js");

    // npm install
    let status = Command::new("npm")
        .arg("install")
        .current_dir(&shim_js_dir)
        .status()
        .expect("failed to run npm install");
    assert!(status.success(), "npm install failed");

    // npm run build
    let status = Command::new("npm")
        .arg("run")
        .arg("build:ts")
        .current_dir(&shim_js_dir)
        .status()
        .expect("failed to run npm run build");
    assert!(status.success(), "npm run build failed");

    println!(
        "cargo:rerun-if-changed={}/package.json",
        shim_js_dir.display()
    );
    println!(
        "cargo:rerun-if-changed={}/package-lock.json",
        shim_js_dir.display()
    );
    println!(
        "cargo:rerun-if-changed={}/src/index.ts",
        shim_js_dir.display()
    );
}
