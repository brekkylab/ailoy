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
    use cmake::Config;

    // Setup directories
    let cargo_manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let cmake_source_dir = cargo_manifest_dir.join("shim");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let cmake_install_dir = out_dir.parent().unwrap().join("deps");

    // Forward LD_LIBRARY_PATH if provided
    if let Ok(ld_path) = std::env::var("LD_LIBRARY_PATH") {
        // Split and add each path separately
        #[cfg(target_os = "windows")]
        let lib_separator = ';';
        #[cfg(not(target_os = "windows"))]
        let lib_separator = ':';

        for path in ld_path.split(lib_separator) {
            if !path.is_empty() {
                println!("cargo:rustc-link-search=native={}", path);
            }
        }
    }

    // Run CMake
    let mut cmake_config = Config::new(&cmake_source_dir);
    cmake_config.profile("Release"); // Always build Release
    cmake_config.define("CMAKE_INSTALL_PREFIX", &cmake_install_dir);

    // Add OpenMP_ROOT if macos
    #[cfg(target_os = "macos")]
    {
        cmake_config.define("MACOSX_DEPLOYMENT_TARGET", "14.0");

        // Link OpenMP(brew installed)
        let libomp_path = std::process::Command::new("brew")
            .arg("--prefix")
            .arg("libomp")
            .output()
            .expect("Failed to execute brew command")
            .stdout;
        let mut libomp_path_str = String::from_utf8(libomp_path).unwrap();
        libomp_path_str = libomp_path_str.trim().to_string();
        cmake_config.define("OpenMP_ROOT", libomp_path_str.clone());

        println!("cargo:rustc-link-search=native={}/lib", libomp_path_str);
        println!("cargo:rustc-link-lib=omp");
    }

    cmake_config.build();

    // Link to this project
    println!(
        "cargo:rustc-link-search=native={}",
        cmake_install_dir.display()
    );
    println!("cargo:rustc-link-lib=static=ailoy_cpp_shim");

    // Link libfaiss.a
    println!("cargo:rustc-link-lib=static=faiss");

    #[cfg(target_os = "linux")]
    {
        // manylinux uses libstdc++.so
        println!("cargo:rustc-link-lib=dylib=stdc++");

        // Linux/ELF: ... -Wl,--whole-archive -l:libtvm_runtime.a -l:libtvm_ffi_static.a -Wl,--no-whole-archive
        println!("cargo:rustc-link-arg=-Wl,--whole-archive");
        println!("cargo:rustc-link-arg=-Wl,-l:libtvm_runtime.a");
        println!("cargo:rustc-link-arg=-Wl,-l:libtvm_ffi_static.a");
        println!("cargo:rustc-link-arg=-Wl,--no-whole-archive");

        // Link FAISS dependencies
        println!("cargo:rustc-link-lib=gomp"); // GNU OpenMP
        println!("cargo:rustc-link-lib=static=openblas"); // OpenBLAS
        println!("cargo:rustc-link-lib=static=gfortran"); // gfortran (required by OpenBLAS)

        // Link Vulkan
        println!("cargo:rustc-link-lib=dylib=vulkan");
    }
    #[cfg(target_os = "macos")]
    {
        // Link Apple Clang c++
        println!("cargo:rustc-link-lib=c++");

        // macOS: ... -Wl,-force_load,/abs/path/to/libtvm_runtime.a
        println!(
            "cargo:rustc-link-arg=-Wl,-force_load,{}",
            (cmake_install_dir.join("libtvm_runtime.a")).display()
        );
        // macOS: ... -Wl,-force_load,/abs/path/to/libtvm_ffi_static.a
        println!(
            "cargo:rustc-link-arg=-Wl,-force_load,{}",
            (cmake_install_dir.join("libtvm_ffi_static.a")).display()
        );

        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }
    #[cfg(target_os = "windows")]
    {
        // Windows (MSVC): ... /WHOLEARCHIVE:tvm_runtime.lib
        println!(
            "cargo:rustc-link-arg=/WHOLEARCHIVE:{}",
            (cmake_install_dir.join("tvm_runtime.lib")).display()
        );
        // Windows (MSVC): ... /WHOLEARCHIVE:tvm_ffi_static.lib
        println!(
            "cargo:rustc-link-arg=/WHOLEARCHIVE:{}",
            (cmake_install_dir.join("tvm_ffi_static.lib")).display()
        );

        // Link FAISS dependencies
        println!("cargo:rustc-link-lib=static=blas");
        println!("cargo:rustc-link-lib=static=lapack");
        println!("cargo:rustc-link-lib=libomp");

        // Link Vulkan
        println!("cargo:rustc-link-lib=dylib=vulkan-1");
    }

    if std::env::var_os("CARGO_FEATURE_NODEJS").is_some() {
        napi_build::setup();
    }

    println!(
        "cargo:rerun-if-changed={}",
        cmake_source_dir.join("CMakeLists.txt").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        cmake_source_dir.join("src/*").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        cmake_source_dir.join("include/*").display()
    );
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
