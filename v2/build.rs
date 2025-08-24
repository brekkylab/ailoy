use std::{env, path::PathBuf};

fn main() {
    // Set target triple
    let target = std::env::var("TARGET").expect("TARGET not set");
    println!("cargo:rustc-env=BUILD_TARGET_TRIPLE={}", target);

    if target.starts_with("wasm") {
        build_wasm();
        return;
    } else {
        build_native();
        return;
    }
}

fn build_native() {
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

fn build_wasm() {
    use std::process::Command;

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
        .arg("build")
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
