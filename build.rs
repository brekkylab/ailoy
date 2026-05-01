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
    if std::env::var_os("CARGO_FEATURE_NODEJS").is_some() {
        napi_build::setup();
    }

    // Link libmlc_llm_module so its TVM_FFI_STATIC_INIT_BLOCKs run at process
    // load. We deliberately link the *_module variant rather than libmlc_llm:
    // mlc-llm's Python package loads libmlc_llm_module via tvm.ffi.load_module,
    // so linking the same dylib keeps a single GlobalFunctionTable in process
    // when both ailoy and mlc-llm are imported together. Linking libmlc_llm
    // directly would put two copies of every mlc.json_ffi.* function in the
    // global registry and the second registration aborts at process load.
    println!("cargo:rerun-if-env-changed=MLC_LLM_LIB_DIR");
    if let Ok(mlc_dir) = std::env::var("MLC_LLM_LIB_DIR") {
        println!("cargo:rustc-link-search=native={}", mlc_dir);
        println!("cargo:rustc-link-lib=dylib=mlc_llm_module");
    }

    // Make the resulting cdylib portable: tell the linker to look for runtime
    // dependencies right next to itself (`@loader_path`). Combined with the
    // build/install step that copies libmlc_llm_module / libtvm{,_runtime,_ffi
    // [_testing]} into bindings/python/ailoy/, this gives ailoy a self-
    // contained dylib closure with no hard-coded venv paths.
    println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path");
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
