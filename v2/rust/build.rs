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
    use std::{env, fs::read_dir, path::PathBuf, process::Command};

    use cmake::Config;

    fn find_cargo_target_dir() -> PathBuf {
        if let Ok(dir) = env::var("CARGO_TARGET_DIR") {
            PathBuf::from(dir)
        } else {
            let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
            for ancestor in out_dir.ancestors() {
                if ancestor.ends_with("build") {
                    return ancestor
                        .parent() // debug or release
                        .and_then(|p| p.parent()) // target or custom root
                        .expect("Invalid OUT_DIR structure")
                        .to_path_buf();
                }
            }
            panic!("Failed to determine target dir from OUT_DIR");
        }
    }

    // Setup directories
    let cargo_manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let cmake_source_dir = cargo_manifest_dir.join("../cpp");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let cargo_target_dir = find_cargo_target_dir();
    let cmake_binary_dir = out_dir.join("build");

    // Download TVM if not exists
    let tvm_dir = cmake_source_dir.join("3rdparty").join("tvm");
    if !tvm_dir.exists() {
        std::fs::create_dir_all(tvm_dir.parent().unwrap()).unwrap();
        let status = Command::new("git")
            .args([
                "clone",
                "--recursive",
                "--depth",
                "1",
                "https://github.com/brekkylab/relax",
                tvm_dir.to_str().unwrap(),
            ])
            .status()
            .expect("Failed to run git clone for TVM");
        if !status.success() {
            panic!("Git clone of TVM failed");
        }
    } else {
        println!("cargo:warning=TVM already exists. Skipping clone.");
    }

    // Run cxx bridge
    cxx_build::bridge("src/ffi/cxx_bridge.rs")
        .file(cmake_source_dir.join("src").join("rust_bridge.cpp"))
        .file(cmake_source_dir.join("src").join("cache.cpp"))
        .flag_if_supported("-std=c++20")
        .include(&cmake_source_dir.join("include"))
        .include(
            &cmake_source_dir
                .join("3rdparty")
                .join("tvm")
                .join("3rdparty")
                .join("dlpack")
                .join("include"),
        )
        .include(cargo_target_dir.join("cxxbridge"))
        .compile("rust_bridge");

    // Run CMake
    Config::new(&cmake_source_dir)
        .env("CARGO_TARGET_DIR", &cargo_target_dir)
        .define("TVM_ROOT", &tvm_dir)
        .define(
            "CMAKE_INSTALL_PREFIX",
            &out_dir
                .parent()
                .unwrap()
                .parent()
                .unwrap()
                .parent()
                .unwrap()
                .join("deps"),
        )
        .build();
    Command::new("cmake")
        .arg("--install")
        .arg(&cmake_binary_dir)
        .status()
        .expect("failed to run cmake install");

    // Link to this project
    println!("cargo:rustc-link-lib=c++");
    println!(
        "cargo:rustc-link-search=native={}",
        out_dir
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("deps")
            .display()
    );
    println!("cargo:rustc-link-lib=static=ailoy_cpp");
    println!("cargo:rustc-link-lib=dylib=tvm_runtime");
    println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path");

    // Add rerun targets
    let include_files = read_dir(&cmake_source_dir.join("include"))
        .expect("Failed to read cpp/include")
        .filter_map(Result::ok);
    let cpp_files = read_dir(&cmake_source_dir.join("src"))
        .expect("Failed to read cpp/src")
        .filter_map(Result::ok);
    let source_files = include_files
        .chain(cpp_files)
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .map(|ext| ext == "cpp" || ext == "hpp")
                .unwrap_or(false)
        });
    for f in source_files {
        println!("cargo:rerun-if-changed={}", f.display());
    }
    println!(
        "cargo:rerun-if-changed={}",
        cmake_source_dir.join("CMakeLists.txt").display()
    );
}
