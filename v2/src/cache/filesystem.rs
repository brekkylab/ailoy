#[cfg(any(target_family = "unix", target_family = "windows"))]
mod native {
    use std::path::Path;
    use tokio::fs::{
        create_dir_all as tokio_create_dir_all, read as tokio_read,
        remove_dir_all as tokio_remove_dir, remove_file as tokio_remove_file, write as tokio_write,
    };

    pub async fn _exists<P: AsRef<Path>>(path: P) -> bool {
        path.as_ref().exists()
    }

    pub async fn read(path: impl AsRef<Path>) -> Result<Vec<u8>, String> {
        tokio_read(path)
            .await
            .map_err(|e| format!("tokio::fs::read failed: {}", e.to_string()))
    }

    pub async fn write(
        path: impl AsRef<Path>,
        data: impl AsRef<[u8]>,
        create_parent: bool,
    ) -> Result<(), String> {
        let parent_dir = path.as_ref().parent().unwrap();
        if !parent_dir.exists() && create_parent {
            tokio_create_dir_all(parent_dir).await.unwrap();
        }
        tokio_write(path, data)
            .await
            .map_err(|e| format!("tokio::fs::write failed: {}", e.to_string()))
    }

    pub async fn _remove(path: impl AsRef<Path>) -> Result<(), String> {
        if path.as_ref().is_dir() {
            tokio_remove_dir(path)
                .await
                .map_err(|e| format!("tokio::fs::remove_dir_all failed: {}", e.to_string()))
        } else if path.as_ref().is_file() {
            tokio_remove_file(path)
                .await
                .map_err(|e| format!("tokio::fs::remove_file failed: {}", e.to_string()))
        } else {
            Err(format!(
                "Neither directory nor file: {}",
                path.as_ref().as_os_str().to_string_lossy()
            ))
        }
    }
}

#[cfg(target_family = "wasm")]
mod opfs {
    use std::path::{Component, Path};

    use js_sys::Uint8Array;
    use wasm_bindgen::JsCast as _;
    use wasm_bindgen_futures::JsFuture;
    use web_sys::{
        FileSystemDirectoryHandle, FileSystemFileHandle, FileSystemGetFileOptions,
        FileSystemRemoveOptions, FileSystemWritableFileStream,
    };

    async fn get_dir_handle(
        path: &Path,
        create: bool,
    ) -> Result<FileSystemDirectoryHandle, String> {
        if path.parent().is_none() {
            return Err("Root is disallowed".to_owned());
        }

        // Initialize `handle` with OPFS root
        let promise = web_sys::window()
            .and_then(|w| Some(w.navigator().storage()))
            .ok_or("no navigator.storage")?
            .get_directory();
        let mut handle = JsFuture::from(promise)
            .await
            .map_err(|_| "Failed to get OPFS root directory")?
            .dyn_into::<FileSystemDirectoryHandle>()
            .map_err(|_| "Invalid root directory handle")?;

        // Call browser API `FileSystemDirectoryHandle` until the parent directory
        for component in path.parent().unwrap().components() {
            if component == Component::RootDir {
                continue;
            }
            let component = component.as_os_str().to_string_lossy();
            let opts = web_sys::FileSystemGetDirectoryOptions::new();
            opts.set_create(create);
            let promise = handle.get_directory_handle_with_options(&component, &opts);
            handle = JsFuture::from(promise)
                .await
                .map_err(|err| {
                    format!(
                        "`FileSystemDirectoryHandle::GetDirectoryHandle failed`: {:?}",
                        err
                    )
                })?
                .dyn_into::<FileSystemDirectoryHandle>()
                .map_err(|_| "Internal error(FileSystemDirectoryHandle)")?;
        }
        Ok(handle)
    }

    async fn get_file_handle(path: &Path, create: bool) -> Result<FileSystemFileHandle, String> {
        let dir_handle = get_dir_handle(path, create).await?;

        // Call browser API `FileSystemDirectoryHandle::get_file_handle`
        let opts = FileSystemGetFileOptions::new();
        opts.set_create(create);
        let promise = dir_handle
            .get_file_handle_with_options(&path.file_name().unwrap().to_string_lossy(), &opts);
        let handle = JsFuture::from(promise)
            .await
            .map_err(|err| {
                format!(
                    "`FileSystemDirectoryHandle::GetFileHandle failed`: {:?}",
                    err
                )
            })?
            .dyn_into::<FileSystemFileHandle>()
            .map_err(|_| "Invalid file handle")?;
        Ok(handle)
    }

    pub async fn exists<P: AsRef<Path>>(path: P) -> bool {
        match get_file_handle(path.as_ref(), false).await {
            Ok(_) => true,
            Err(_) => return false,
        }
    }

    pub async fn read(path: impl AsRef<Path>) -> Result<Vec<u8>, String> {
        // Get file handle
        let handle = get_file_handle(path.as_ref(), false).await?;

        // Call browser API `FileSystemFileHandle::getFile()`
        let promise = handle.get_file();
        let file = JsFuture::from(promise)
            .await
            .map_err(|_| "Failed to get File object")?
            .dyn_into::<web_sys::File>()
            .map_err(|_| "Invalid File object")?;

        // Call browser API `File::array_buffer()`
        let promise = file.array_buffer();
        let array_buffer = JsFuture::from(promise)
            .await
            .map_err(|_| "Failed to read file as ArrayBuffer")?;

        // Convert to Uint8Array
        let uint8_array = Uint8Array::new(&array_buffer);
        let mut vec = vec![0u8; uint8_array.length() as usize];
        uint8_array.copy_to(&mut vec[..]);

        Ok(vec)
    }

    pub async fn write(path: impl AsRef<Path>, data: impl AsRef<[u8]>) -> Result<(), String> {
        // Get file handle
        let handle = get_file_handle(path.as_ref(), true).await?;

        // Create writable stream
        let promise = handle.create_writable();
        let stream = JsFuture::from(promise)
            .await
            .map_err(|_| "Failed to create writable stream")?
            .dyn_into::<FileSystemWritableFileStream>()
            .map_err(|_| "Invalid writable stream")?;

        // Write to file
        let promise = stream
            .write_with_u8_array(data.as_ref())
            .map_err(|_| "`write_with_u8_array` failed")?;
        JsFuture::from(promise)
            .await
            .map_err(|_| "Failed to write to file")?;

        // Close stream
        let close_promise = stream.close();
        JsFuture::from(close_promise)
            .await
            .map_err(|_| "Failed to close file")?;

        Ok(())
    }

    pub async fn remove(path: impl AsRef<Path>) -> Result<(), String> {
        let handle = get_dir_handle(path.as_ref(), false).await?;

        let opts = FileSystemRemoveOptions::new();
        opts.set_recursive(true);
        let name = path.as_ref().file_name().unwrap().to_string_lossy();
        let promise = handle.remove_entry_with_options(&name, &opts);
        JsFuture::from(promise)
            .await
            .map_err(|err| format!("FileSystemDirectoryHandle::remove_entry failed: {:?}", err))?;
        Ok(())
    }

    #[wasm_bindgen::prelude::wasm_bindgen(js_name = "ailoy_filesystem_exists")]
    pub async fn exists_(path: String) -> bool {
        exists(&path).await
    }

    #[wasm_bindgen::prelude::wasm_bindgen(js_name = "ailoy_filesystem_read")]
    pub async fn read_(path: String) -> Result<Vec<u8>, String> {
        read(&path).await
    }

    #[wasm_bindgen::prelude::wasm_bindgen(js_name = "ailoy_filesystem_write")]
    pub async fn write_(path: String, data: js_sys::Uint8Array) -> Result<(), String> {
        write(&path, &data.to_vec()).await
    }

    #[wasm_bindgen::prelude::wasm_bindgen(js_name = "ailoy_filesystem_remove")]
    pub async fn remove_(path: String) -> Result<(), String> {
        remove(&path).await
    }
}

#[cfg(any(target_family = "unix", target_family = "windows"))]
pub use native::*;

#[cfg(target_family = "wasm")]
pub use opfs::*;
