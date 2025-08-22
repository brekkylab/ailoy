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
        return;
    } else {
        build_native();
        return;
    }
}

fn build_native() {
    use std::{env, path::PathBuf};

    use cmake::Config;

    // Setup directories
    let cargo_manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let cmake_source_dir = cargo_manifest_dir.join("shim");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let cmake_install_dir = out_dir.parent().unwrap().join("deps");

    // Run CMake
    Config::new(&cmake_source_dir)
        .define("CMAKE_INSTALL_PREFIX", &cmake_install_dir)
        .build();

    // Link to this project
    println!("cargo:rustc-link-lib=c++");
    println!(
        "cargo:rustc-link-search=native={}",
        cmake_install_dir.display()
    );
    println!("cargo:rustc-link-lib=static=ailoy_cpp_shim");

    #[cfg(target_os = "linux")]
    {
        // Linux/ELF: ... -Wl,--whole-archive -l:libtvm_runtime.a -Wl,--no-whole-archive
        println!("cargo:rustc-link-arg=-Wl,--whole-archive");
        println!("cargo:rustc-link-arg=-Wl,-l:libtvm_runtime.a");
        println!("cargo:rustc-link-arg=-Wl,--no-whole-archive");
    }
    #[cfg(target_os = "macos")]
    {
        // macOS: ... -Wl,-force_load,/abs/path/to/libtvm_runtime.a
        println!(
            "cargo:rustc-link-arg=-Wl,-force_load,{}",
            (cmake_install_dir.join("libtvm_runtime.a")).display()
        );
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=Foundation");
    }
    #[cfg(target_os = "windows")]
    {
        // Windows (MSVC): ... tvm_runtime.lib /WHOLEARCHIVE:tvm_runtime.lib
        println!("cargo:rustc-link-lib=static=tvm_runtime.lib");
        println!(
            "cargo:rustc-link-arg=/WHOLEARCHIVE:{}",
            (cmake_install_dir.join("tvm_runtime.lib")).display()
        );
    }

    if std::env::var_os("CARGO_FEATURE_NODE").is_some() {
        napi_build::setup();
    }

    println!(
        "cargo:rerun-if-changed={}",
        cmake_source_dir.join("CMakeLists.txt").display()
    );
}
