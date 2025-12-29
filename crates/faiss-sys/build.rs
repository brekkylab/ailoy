use std::path::PathBuf;

fn main() {
    let target = std::env::var("TARGET").expect("TARGET not set");
    if target.starts_with("wasm") {
        println!("cargo:warning=ailoy-faiss-sys is not used in WASM target");
        return;
    }

    // Forward LD_LIBRARY_PATH if provided
    let ld_paths = if let Ok(ld_path) = std::env::var("LD_LIBRARY_PATH") {
        #[cfg(target_os = "windows")]
        let lib_separator = ';';
        #[cfg(not(target_os = "windows"))]
        let lib_separator = ':';

        let mut paths = vec![];
        for path in ld_path.split(lib_separator) {
            if !path.is_empty() {
                paths.push(path.to_string());
            }
        }
        paths
    } else {
        vec![]
    };

    let faiss_build_dir = scratch::path("faiss");
    let faiss_libdir = faiss_build_dir.join("lib");
    let faiss_includedir = faiss_build_dir.join("include");

    // Check if Faiss is already built
    let faiss_dylib_path = if cfg!(target_os = "windows") {
        let faiss_bindir = faiss_build_dir.join("bin");
        faiss_bindir.join("faiss.dll")
    } else if cfg!(target_os = "macos") {
        faiss_libdir.join("libfaiss.dylib")
    } else {
        faiss_libdir.join("libfaiss.so")
    };
    if !faiss_dylib_path.exists() {
        println!("cargo:warning=Building Faiss (this may take a while, but will be cached)...");

        let mut cfg = cmake::Config::new("faiss");
        cfg.define("CMAKE_BUILD_TYPE", "Release")
            .define("CMAKE_INSTALL_LIBDIR", "lib")
            .define("BUILD_SHARED_LIBS", "ON")
            .define("BUILD_TESTING", "OFF")
            .define("FAISS_ENABLE_GPU", "OFF")
            .define("FAISS_ENABLE_PYTHON", "OFF")
            .define("FAISS_ENABLE_EXTRAS", "OFF")
            .out_dir(&faiss_build_dir)
            .very_verbose(true);

        // Configure BLAS and LAPACK
        #[cfg(not(target_os = "windows"))]
        {
            for ld_path in ld_paths.iter() {
                let path = PathBuf::from(ld_path);
                if path.join("libblas.a").exists() {
                    cfg.define("BLAS_LIBRARIES", path.join("libblas.a"));
                }
                if path.join("liblapack.a").exists() {
                    cfg.define("LAPACK_LIBRARIES", path.join("liblapack.a"));
                }
            }
        }
        #[cfg(target_os = "windows")]
        {
            for ld_path in ld_paths.iter() {
                let path = PathBuf::from(ld_path);
                if path.join("blas.lib").exists() {
                    cfg.define("BLAS_LIBRARIES", path.join("blas.lib"));
                }
                if path.join("lapack.lib").exists() {
                    cfg.define("LAPACK_LIBRARIES", path.join("lapack.lib"));
                }
            }
        }

        // Configure libomp for macos
        #[cfg(target_os = "macos")]
        {
            cfg.define("OpenMP_ROOT", get_libomp_path());
        }

        cfg.build();
    } else {
        println!(
            "cargo:warning=Using cached Faiss build from: {}",
            faiss_build_dir.display()
        );
    }

    println!("cargo:rustc-link-search=native={}", faiss_libdir.display());
    println!("cargo:rustc-link-lib=faiss");
    for ld_path in ld_paths {
        println!("cargo:rustc-link-search=native={}", ld_path);
    }

    cxx_build::bridge("src/lib.rs")
        .include(&faiss_includedir)
        .file("src/bridge.cpp")
        .std("c++20")
        .compile("cxxbridge-faiss");

    // Patch link path of libfaiss from rpath to absolute path
    #[cfg(target_os = "macos")]
    {
        if faiss_dylib_path.exists() {
            if let Err(e) = std::process::Command::new("install_name_tool")
                .arg("-id")
                .arg(&faiss_dylib_path)
                .arg(&faiss_dylib_path)
                .status()
            {
                println!("cargo:warning=Failed to patch libfaiss.dylib: {:?}", e);
            }
        }
    }
    #[cfg(target_os = "linux")]
    {
        if faiss_dylib_path.exists() {
            if let Err(e) = std::process::Command::new("patchelf")
                .arg("--set-soname")
                .arg(&faiss_dylib_path)
                .arg(&faiss_dylib_path)
                .status()
            {
                println!("cargo:warning=Failed to patch libfaiss.so: {:?}", e);
            }
        }
    }
    #[cfg(target_os = "windows")]
    {
        if faiss_dylib_path.exists() {
            let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
            // Copy to OUT_DIR
            if let Err(e) = std::fs::copy(&faiss_dylib_path, out_dir.join("faiss.dll")) {
                println!("cargo:warning=Failed to copy faiss.dll to OUT_DIR: {:?}", e);
            }

            // Try to copy to the target/profile directory and deps folder
            let target_dir = out_dir.clone();
            if let Some(build_dir_idx) = target_dir
                .components()
                .enumerate()
                .find(|(_, c)| c.as_os_str() == "build")
                .map(|(i, _)| i)
            {
                let mut path = PathBuf::new();
                for (i, c) in target_dir.components().enumerate() {
                    if i >= build_dir_idx {
                        break;
                    }
                    path.push(c);
                }
                // path is now target/debug
                if let Err(e) = std::fs::copy(&faiss_dylib_path, path.join("faiss.dll")) {
                    println!("cargo:warning=Failed to copy faiss.dll to target build directory");
                }
                if let Err(e) =
                    std::fs::copy(&faiss_dylib_path, path.join("deps").join("faiss.dll"))
                {
                    println!(
                        "cargo:warning=Failed to copy faiss.dll to `deps` under target build directory"
                    );
                }
            }
        }
    }

    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=faiss");
}

#[cfg(target_os = "macos")]
fn get_libomp_path() -> String {
    let libomp_path = std::process::Command::new("brew")
        .arg("--prefix")
        .arg("libomp")
        .output()
        .expect("Failed to execute brew command")
        .stdout;

    let mut libomp_path = String::from_utf8(libomp_path).unwrap();
    libomp_path = libomp_path.trim().to_string();
    libomp_path
}
