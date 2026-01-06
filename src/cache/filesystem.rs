#[cfg(any(target_family = "unix", target_family = "windows"))]
mod native {
    use std::path::Path;

    use anyhow::{Context, bail};
    use tokio::fs::{
        create_dir_all as tokio_create_dir_all, read as tokio_read,
        remove_dir_all as tokio_remove_dir, remove_file as tokio_remove_file, write as tokio_write,
    };

    pub async fn _exists<P: AsRef<Path>>(path: P) -> bool {
        path.as_ref().exists()
    }

    pub async fn read(path: impl AsRef<Path>) -> anyhow::Result<Vec<u8>> {
        tokio_read(path).await.context("tokio::fs::read failed")
    }

    pub async fn write(
        path: impl AsRef<Path>,
        data: impl AsRef<[u8]>,
        create_parent: bool,
    ) -> anyhow::Result<()> {
        let parent_dir = path.as_ref().parent().unwrap();
        if !parent_dir.exists() && create_parent {
            tokio_create_dir_all(parent_dir)
                .await
                .context("tokio::fs::create_dir_all failed")?;
        }
        tokio_write(path, data)
            .await
            .context("tokio::fs::write failed")
    }

    pub async fn remove(path: impl AsRef<Path>) -> anyhow::Result<()> {
        if path.as_ref().is_dir() {
            tokio_remove_dir(path)
                .await
                .context("tokio::fs::remove_dir_all failed")
        } else if path.as_ref().is_file() {
            tokio_remove_file(path)
                .await
                .context("tokio::fs::remove_file failed")
        } else {
            bail!(
                "Neither directory nor file: {}",
                path.as_ref().as_os_str().to_string_lossy()
            )
        }
    }
}

#[cfg(target_family = "wasm")]
mod opfs {
    use std::path::{Component, Path};

    use anyhow::{anyhow, bail};
    use js_sys::Uint8Array;
    use wasm_bindgen::{JsCast as _, prelude::*};
    use wasm_bindgen_futures::JsFuture;
    use web_sys::{
        FileSystemDirectoryHandle, FileSystemFileHandle, FileSystemGetFileOptions,
        FileSystemRemoveOptions, FileSystemWritableFileStream,
    };

    #[wasm_bindgen]
    extern "C" {
        type Global;

        #[wasm_bindgen(method, getter, js_name = Window)]
        fn window(this: &Global) -> JsValue;

        #[wasm_bindgen(method, getter, js_name = WorkerGlobalScope)]
        fn worker(this: &Global) -> JsValue;
    }

    async fn get_dir_handle(
        path: &Path,
        create: bool,
    ) -> anyhow::Result<FileSystemDirectoryHandle> {
        if path.parent().is_none() {
            bail!("Root is disallowed".to_owned());
        }

        let global: Global = js_sys::global().unchecked_into();
        let storage = if !global.window().is_undefined() {
            global
                .unchecked_into::<web_sys::Window>()
                .navigator()
                .storage()
        } else if !global.worker().is_undefined() {
            global
                .unchecked_into::<web_sys::WorkerGlobalScope>()
                .navigator()
                .storage()
        } else {
            bail!("Failed to get navigator.storage");
        };

        // Initialize `handle` with OPFS root
        let mut handle = JsFuture::from(storage.get_directory())
            .await
            .map_err(|_| anyhow!("Failed to get OPFS root directory"))?
            .dyn_into::<FileSystemDirectoryHandle>()
            .map_err(|_| anyhow!("Invalid root directory handle"))?;

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
                .map_err(|_| anyhow!("`FileSystemDirectoryHandle::GetDirectoryHandle failed`"))?
                .dyn_into::<FileSystemDirectoryHandle>()
                .map_err(|_| anyhow!("Internal error(FileSystemDirectoryHandle)"))?;
        }
        Ok(handle)
    }

    async fn get_file_handle(path: &Path, create: bool) -> anyhow::Result<FileSystemFileHandle> {
        let dir_handle = get_dir_handle(path, create).await?;

        // Call browser API `FileSystemDirectoryHandle::get_file_handle`
        let opts = FileSystemGetFileOptions::new();
        opts.set_create(create);
        let promise = dir_handle
            .get_file_handle_with_options(&path.file_name().unwrap().to_string_lossy(), &opts);
        let handle = JsFuture::from(promise)
            .await
            .map_err(|_| anyhow!("`FileSystemDirectoryHandle::GetFileHandle failed`"))?
            .dyn_into::<FileSystemFileHandle>()
            .map_err(|_| anyhow!("Invalid file handle"))?;
        Ok(handle)
    }

    pub async fn _exists<P: AsRef<Path>>(path: P) -> bool {
        match get_file_handle(path.as_ref(), false).await {
            Ok(_) => true,
            Err(_) => return false,
        }
    }

    pub async fn read(path: impl AsRef<Path>) -> anyhow::Result<Vec<u8>> {
        // Get file handle
        let handle = get_file_handle(path.as_ref(), false).await?;

        // Call browser API `FileSystemFileHandle::getFile()`
        let promise = handle.get_file();
        let file = JsFuture::from(promise)
            .await
            .map_err(|_| anyhow!("Failed to get File object"))?
            .dyn_into::<web_sys::File>()
            .map_err(|_| anyhow!("Invalid File object"))?;

        // Call browser API `File::array_buffer()`
        let promise = file.array_buffer();
        let array_buffer = JsFuture::from(promise)
            .await
            .map_err(|_| anyhow!("Failed to read file as ArrayBuffer"))?;

        // Convert to Uint8Array
        let uint8_array = Uint8Array::new(&array_buffer);
        let mut vec = vec![0u8; uint8_array.length() as usize];
        uint8_array.copy_to(&mut vec[..]);

        Ok(vec)
    }

    pub async fn write(
        path: impl AsRef<Path>,
        data: impl AsRef<[u8]>,
        create_parent: bool,
    ) -> anyhow::Result<()> {
        // Get file handle
        let handle = get_file_handle(path.as_ref(), create_parent).await?;

        // Create writable stream
        let promise = handle.create_writable();
        let stream = JsFuture::from(promise)
            .await
            .map_err(|_| anyhow!("Failed to create writable stream"))?
            .dyn_into::<FileSystemWritableFileStream>()
            .map_err(|_| anyhow!("Invalid writable stream"))?;

        // Write to file
        let promise = stream
            .write_with_u8_array(data.as_ref())
            .map_err(|_| anyhow!("`write_with_u8_array` failed"))?;
        JsFuture::from(promise)
            .await
            .map_err(|e| anyhow!("Failed to write to file: {:?}", e.as_string()))?;

        // Close stream
        let close_promise = stream.close();
        JsFuture::from(close_promise)
            .await
            .map_err(|_| anyhow!("Failed to close file"))?;

        Ok(())
    }

    pub async fn remove(path: impl AsRef<Path>) -> anyhow::Result<()> {
        let handle = get_dir_handle(path.as_ref(), false).await?;

        let opts = FileSystemRemoveOptions::new();
        opts.set_recursive(true);
        let name = path.as_ref().file_name().unwrap().to_string_lossy();
        let promise = handle.remove_entry_with_options(&name, &opts);
        JsFuture::from(promise)
            .await
            .map_err(|_| anyhow!("FileSystemDirectoryHandle::remove_entry failed"))?;
        Ok(())
    }
}

#[cfg(any(target_family = "unix", target_family = "windows"))]
pub use native::*;
#[cfg(target_family = "wasm")]
pub use opfs::*;
