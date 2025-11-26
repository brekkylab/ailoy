use std::path::PathBuf;

fn main() {
    let target = std::env::var("TARGET").expect("TARGET not set");
    if target.starts_with("wasm") {
        println!("cargo:warning=ailoy-faiss-sys is not used in WASM target");
        return;
    }

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

    let out_dir = std::env::var("OUT_DIR").map(PathBuf::from).unwrap();
    let faiss_build_dir = out_dir.join("faiss");
    let faiss_libdir = faiss_build_dir.join("lib");
    let faiss_includedir = faiss_build_dir.join("include");

    // Check if Faiss is already built
    let faiss_lib = if cfg!(target_os = "windows") {
        faiss_libdir.join("faiss.lib")
    } else {
        faiss_libdir.join("libfaiss.a")
    };
    if !faiss_lib.exists() {
        println!("cargo:warning=Building Faiss (this may take a while, but will be cached)...");

        let mut cfg = cmake::Config::new("faiss");
        cfg.define("CMAKE_BUILD_TYPE", "Release")
            .define("CMAKE_INSTALL_LIBDIR", "lib")
            .define("BUILD_SHARED_LIBS", "OFF")
            .define("BUILD_TESTING", "OFF")
            .define("FAISS_ENABLE_GPU", "OFF")
            .define("FAISS_ENABLE_PYTHON", "OFF")
            .define("FAISS_ENABLE_EXTRAS", "OFF")
            .out_dir(&faiss_build_dir)
            .very_verbose(true);

        #[cfg(not(target_os = "windows"))]
        {
            cfg.define("BLAS_LIBRARIES", "libblas.a")
                .define("LAPACK_LIBRARIES", "liblapack.a");
        }
        #[cfg(target_os = "windows")]
        {
            cfg.define("BLAS_LIBRARIES", "blas.lib")
                .define("LAPACK_LIBRARIES", "lapack.lib");
        }

        cfg.build();
    } else {
        println!(
            "cargo:warning=Using cached Faiss build from: {}",
            faiss_build_dir.display()
        );
    }

    // Link OpenMP
    #[cfg(target_os = "macos")]
    {
        // Link OpenMP(brew installed)
        let libomp_path = std::process::Command::new("brew")
            .arg("--prefix")
            .arg("libomp")
            .output()
            .expect("Failed to execute brew command")
            .stdout;

        let mut libomp_path = String::from_utf8(libomp_path).unwrap();
        libomp_path = libomp_path.trim().to_string();
        cfg.define("OpenMP_ROOT", libomp_path.clone());

        println!("cargo:rustc-link-search=native={}/lib", libomp_path);
        println!("cargo:rustc-link-lib=omp");
    }
    #[cfg(target_os = "linux")]
    {
        // Link GNU OpenMP
        println!("cargo:rustc-link-lib=gomp");
    }
    #[cfg(target_os = "windows")]
    {
        // Link MSVC OpenMP
        println!("cargo:rustc-link-lib=static=libomp");
    }

    println!("cargo:rustc-link-search=native={}", faiss_libdir.display());
    println!("cargo:rustc-link-lib=static=faiss");
    println!("cargo:rustc-link-lib=blas");
    println!("cargo:rustc-link-lib=lapack");

    cxx_build::bridge("src/lib.rs")
        .include(&faiss_includedir)
        .file("src/bridge.cpp")
        .std("c++20")
        .compile("cxxbridge-faiss");

    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=faiss");
}
