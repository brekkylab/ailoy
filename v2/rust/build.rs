use std::{env, fs::create_dir_all, path::PathBuf, process::Command};

fn main() {
    // Set target triple
    let target = env::var("TARGET").expect("TARGET not set");
    println!("cargo:rustc-env=BUILD_TARGET_TRIPLE={}", target);
    let cmake_build_type = match env::var("PROFILE").unwrap().as_str() {
        "debug" => "Debug",
        "release" => "Release",
        _ => "",
    };
    if cmake_build_type == "" {
        panic!(
            "Unsupported build type: {}",
            env::var("PROFILE").unwrap().as_str()
        );
    }

    // CMake build + install paths
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let cmake_source_dir = manifest_dir.join("../cpp"); // Update this
    let cmake_build_dir = manifest_dir.join("build").join(&target);
    if !cmake_build_dir.exists() {
        create_dir_all(&cmake_build_dir).unwrap();
    }

    // Run cmake configure
    let status = Command::new("cmake")
        .arg("-S")
        .arg(&cmake_source_dir)
        .arg("-B")
        .arg(&cmake_build_dir)
        .arg(format!("-DCMAKE_BUILD_TYPE={}", cmake_build_type))
        .status()
        .expect("Failed to cmake configure");
    assert!(status.success(), "CMake configure failed");

    // CMake build
    let status = Command::new("cmake")
        .arg("--build")
        .arg(&cmake_build_dir)
        .arg("--config")
        .arg(cmake_build_type)
        .status()
        .expect("Failed to build with cmake");
    assert!(status.success(), "CMake build failed");

    // Link library
    let lib_dir = cmake_build_dir.clone();
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=static=ailoy_cpp");
    println!("cargo:rustc-link-lib=dylib=tvm_runtime");
    println!(
        "cargo:rustc-link-search=native={}/_deps/tvm-build",
        cmake_build_dir.to_str().unwrap(),
    );
    println!("cargo:rustc-link-lib=c++");

    // Re-run triggers
    println!("cargo:rerun-if-changed=../cpp/CMakeLists.txt");
    println!("cargo:rerun-if-env-changed=TARGET");
    println!("cargo:rerun-if-env-changed=PROFILE");
}
