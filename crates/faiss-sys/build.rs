fn main() {
    let target = std::env::var("TARGET").expect("TARGET not set");
    if target.starts_with("wasm") {
        println!("cargo:warning=ailoy-faiss-sys is not used in WASM target");
        return;
    }

    let mut cfg = cmake::Config::new("faiss");
    cfg.define("CMAKE_BUILD_TYPE", "Release")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("BUILD_TESTING", "OFF")
        .define("FAISS_ENABLE_GPU", "OFF")
        .define("FAISS_ENABLE_PYTHON", "OFF")
        .define("FAISS_ENABLE_EXTRAS", "OFF")
        .very_verbose(true);

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

    let dst = cfg.build();
    let faiss_libdir = dst.join("lib");
    let faiss_includedir = dst.join("include");

    println!("cargo:rustc-link-search=native={}", faiss_libdir.display());
    println!("cargo:rustc-link-lib=static=faiss");
    println!("cargo:rustc-link-lib=blas");
    println!("cargo:rustc-link-lib=lapack");

    let crate_name = std::env::var("CARGO_PKG_NAME").unwrap();
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let out_dir = std::env::var("OUT_DIR").unwrap();

    cxx_build::bridge("src/lib.rs")
        .include(format!("{}/src", manifest_dir))
        .include(&faiss_includedir)
        .include(&out_dir)
        .include(format!("{}/cxxbridge/include/{}/src", out_dir, crate_name))
        .file("src/bridge.cpp")
        .std("c++20")
        .compile("cxxbridge-faiss");
    println!("cargo:rerun-if-changed=src/*.hpp");
    println!("cargo:rerun-if-changed=src/*.cpp");
}
