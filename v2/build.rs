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

    // Link libfaiss.a
    println!(
        "cargo:rustc-link-search=native={}",
        (cmake_install_dir.join("libfaiss.a")).display()
    );
    println!("cargo:rustc-link-lib=static=faiss");

    #[cfg(target_os = "linux")]
    {
        // Linux/ELF: ... -Wl,--whole-archive -l:libtvm_runtime.a -Wl,--no-whole-archive
        println!("cargo:rustc-link-arg=-Wl,--whole-archive");
        println!("cargo:rustc-link-arg=-Wl,-l:libtvm_runtime.a");
        println!("cargo:rustc-link-arg=-Wl,--no-whole-archive");

        // Link FAISS dependencies (not tested)
        println!("cargo:rustc-link-lib=gomp"); // GNU OpenMP
        println!("cargo:rustc-link-lib=blas"); // BLAS
        println!("cargo:rustc-link-lib=lapack"); // LAPACK
    }
    #[cfg(target_os = "macos")]
    {
        // macOS: ... -Wl,-force_load,/abs/path/to/libtvm_runtime.a
        println!(
            "cargo:rustc-link-arg=-Wl,-force_load,{}",
            (cmake_install_dir.join("libtvm_runtime.a")).display()
        );

        // Link OpenMP(brew installed)
        let libomp_path = std::process::Command::new("brew")
            .arg("--prefix")
            .arg("libomp")
            .output()
            .expect("Failed to execute brew command")
            .stdout;
        let libomp_path_str = String::from_utf8(libomp_path).unwrap();
        println!(
            "cargo:rustc-link-search=native={}/lib",
            libomp_path_str.trim()
        );
        println!("cargo:rustc-link-lib=omp");

        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }
    #[cfg(target_os = "windows")]
    {
        // Windows (MSVC): ... tvm_runtime.lib /WHOLEARCHIVE:tvm_runtime.lib
        println!("cargo:rustc-link-lib=static=tvm_runtime.lib");
        println!(
            "cargo:rustc-link-arg=/WHOLEARCHIVE:{}",
            (cmake_install_dir.join("tvm_runtime.lib")).display()
        );

        // Link MKL for FAISS (not tested)
        let mkl_root = env::var("MKL_ROOT")
            .expect("MKL_ROOT environment variable not set. Please set it to your Intel MKL installation path.");

        println!("cargo:rustc-link-search=native={}/lib/intel64", mkl_root);

        // MKL core libraries and Intel OpenMP Runtime
        println!("cargo:rustc-link-lib=static=mkl_intel_lp64");
        println!("cargo:rustc-link-lib=static=mkl_sequential"); // or mkl_tbb_thread
        println!("cargo:rustc-link-lib=static=mkl_core");
        println!("cargo:rustc-link-lib=dylib=libiomp5md");
    }

    if std::env::var_os("CARGO_FEATURE_NODE").is_some() {
        napi_build::setup();
    }

    println!(
        "cargo:rerun-if-changed={}",
        cmake_source_dir.join("CMakeLists.txt").display()
    );
}
