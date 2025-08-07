fn main() {
    // Set target triple
    let target = std::env::var("TARGET").expect("TARGET not set");
    println!("cargo:rustc-env=BUILD_TARGET_TRIPLE={}", target);

    if target.starts_with("wasm") {
        return;
    } else {
        build_native();
        return;
    }
}

fn build_native() {
    use std::{env, path::PathBuf, process::Command};

    use cmake::Config;

    // Set directories
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let cmake_source_dir = manifest_dir.join("../cpp");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let target_dir = out_dir
        .ancestors()
        .nth(3)
        .expect("Failed to determine Rust target directory")
        .to_path_buf();

    // CMake
    let dst = Config::new(&cmake_source_dir)
        .define("CMAKE_INSTALL_PREFIX", &target_dir.join("deps"))
        .build();
    Command::new("cmake")
        .arg("--install")
        .arg(&dst)
        .status()
        .expect("failed to run cmake install");
    println!("cargo:rustc-link-lib=c++");
    println!(
        "cargo:rustc-link-search=native={}",
        target_dir.join("deps").display()
    );
    println!("cargo:rustc-link-lib=static=ailoy_cpp");
    println!("cargo:rustc-link-lib=dylib=tvm_runtime");
    println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path");
    println!("cargo:rerun-if-changed=../cpp/src/tvm_runtime.hpp");
    println!("cargo:rerun-if-changed=../cpp/src/tvm_runtime.cpp");
    println!("cargo:rerun-if-changed=../cpp/src/language_model.hpp");
    println!("cargo:rerun-if-changed=../cpp/src/language_model.cpp");
    println!("cargo:rerun-if-changed=../cpp/CMakeLists.txt");
}
