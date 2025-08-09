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
    use std::{
        env,
        fs::read_dir,
        path::{Path, PathBuf},
        process::Command,
    };

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

    fn clone(name: &str, url: &str, dir: &Path) {
        use std::fs;
        use std::process::Command;

        if dir.exists() {
            println!("cargo:warning={} already exists. Skipping clone.", name);
            return;
        }

        fs::create_dir_all(dir.parent().unwrap()).unwrap();

        let status = Command::new("git")
            .args([
                "clone",
                "--depth",
                "1",
                if name == "tvm" { "--recursive" } else { "" },
                url,
                dir.to_str().unwrap(),
            ])
            .status()
            .expect(&format!("Failed to run git clone for {}", name));

        if !status.success() {
            panic!("Git clone of {} failed", name);
        }
    }

    // Setup directories
    let cargo_manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let cmake_source_dir = cargo_manifest_dir.join("../cpp");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let cargo_target_dir = find_cargo_target_dir();
    let cmake_binary_dir = out_dir.join("build");
    let cmake_install_dir = out_dir
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("deps");
    let cpp_include_files = read_dir(&cmake_source_dir.join("include"))
        .expect("Failed to read cpp/include")
        .filter_map(Result::ok)
        .into_iter()
        .map(|entry| entry.path())
        .filter(|path| path.extension().map(|ext| ext == "hpp").unwrap_or(false))
        .collect::<Vec<_>>();
    let cpp_source_files = read_dir(&cmake_source_dir.join("src"))
        .expect("Failed to read cpp/src")
        .filter_map(Result::ok)
        .into_iter()
        .map(|entry| entry.path())
        .filter(|path| path.extension().map(|ext| ext == "hpp").unwrap_or(false))
        .collect::<Vec<_>>();
    let cpp_files = cpp_include_files
        .clone()
        .into_iter()
        .chain(cpp_source_files.clone().into_iter())
        .collect::<Vec<_>>();

    // Download if not exists
    let tvm_dir = cmake_source_dir.join("3rdparty").join("tvm");
    clone("tvm", "https://github.com/brekkylab/relax", &tvm_dir);
    let json_dir = cmake_source_dir.join("3rdparty").join("json");
    clone("json", "https://github.com/nlohmann/json.git", &json_dir);

    // Run cxx bridge
    let mut cxx = cxx_build::bridge("src/ffi/cxx_bridge.rs");
    for f in &cpp_source_files {
        cxx.file(f);
    }
    cxx.flag_if_supported("-std=c++20")
        .include(&cmake_source_dir.join("include"))
        .include(&tvm_dir.join("include"))
        .include(&tvm_dir.join("ffi").join("include"))
        .include(&tvm_dir.join("3rdparty").join("dlpack").join("include"))
        .include(&tvm_dir.join("3rdparty").join("dmlc-core").join("include"))
        .include(&json_dir.join("include"))
        .include(cargo_target_dir.join("cxxbridge"))
        .compile("rust_bridge");

    // Run CMake
    Config::new(&cmake_source_dir)
        .env("CARGO_TARGET_DIR", &cargo_target_dir)
        .define("TVM_ROOT", &tvm_dir)
        .define("CMAKE_INSTALL_PREFIX", &cmake_install_dir)
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
        cmake_install_dir.display()
    );
    println!("cargo:rustc-link-lib=static=ailoy_cpp");
    println!("cargo:rustc-link-lib=dylib=tvm_runtime");
    println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path");

    // Add rerun targets
    for f in &cpp_files {
        println!("cargo:rerun-if-changed={}", f.display());
    }
    println!(
        "cargo:rerun-if-changed={}",
        cmake_source_dir.join("CMakeLists.txt").display()
    );
}
