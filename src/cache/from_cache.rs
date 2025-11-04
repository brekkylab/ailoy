use std::pin::Pin;

use crate::{
    cache::{Cache, CacheClaim, CacheContents},
    utils::{BoxFuture, MaybeSend},
};

/// A trait for reconstructing values from cached files managed by [`Cache`].
///
/// # Overview
/// Implementors describe how to:
/// 1. **Declare** which files must be fetched (`claim_files`)
/// 2. **Reconstruct** `Self` once the files are available (`try_from_contents`)
///
/// This enables the cache system to automatically resolve, download,
/// and assemble complex structures such as models or embeddings.
///
/// # Workflow
/// 1. **`claim_files(cache, key, ctx)`**
///    - Declare which logical cache entries (`CacheEntry`s) are required.
///    - Optionally populate or inspect the provided `ctx` (a mutable
///      [`HashMap<String, Value>`]) to pass metadata or intermediate data.
/// 2. **Cache layer loads the files**
///    - The caller (typically [`Cache::try_create`]) fetches local or remote files,
///      groups them into a [`CacheContents`], and passes them to the next stage.
/// 3. **`try_from_contents(contents, ctx)`**
///    - Consume the provided [`CacheContents`] and asynchronously reconstruct `Self`.
///    - The same `ctx` reference from `claim_files` is passed, allowing contextual reuse.
///
/// # Context (`ctx`)
/// - `ctx` is a mutable key–value map (`HashMap<String, Value>`) that acts as a
///   lightweight "construction context".
/// - If no shared state is needed, you can ignore it.
///
/// # Contract
/// - `claim_files` must be **lightweight**: it should only declare dependencies,
///   not perform file reads or heavy computation.
/// - Filenames must match the manifest paths (e.g. `"model.safetensors"`).
/// - `try_from_contents` must **validate and consume** required entries, removing
///   them from `contents` as it proceeds.
/// - Avoid unnecessary copies—large models may load multi-gigabyte files.
///
/// # Example
/// ```rust,ignore
/// use crate::cache::{Cache, CacheClaim, CacheContents, TryFromCache};
///
/// struct MyModel { tokenizer: Vec<u8>, weights: Vec<u8> }
///
/// impl TryFromCache for MyModel {
///     fn claim_files<'a>(
///         cache: Cache,
///         key: impl AsRef<str>,
///         ctx: &'a mut std::collections::HashMap<String, crate::value::Value>,
///     ) -> BoxFuture<'a, anyhow::Result<CacheClaim>> {
///         let key = key.as_ref().to_owned();
///         Box::pin(async move {
///             Ok(CacheClaim::new(vec![
///                 (key.clone(), "tokenizer.json"),
///                 (key, "model.safetensors"),
///             ]))
///         })
///     }
///
///     fn try_from_contents<'a>(
///         mut contents: CacheContents,
///         _ctx: &'a std::collections::HashMap<String, crate::value::Value>,
///     ) -> BoxFuture<'a, anyhow::Result<Self>> {
///         Box::pin(async move {
///             let tokenizer = contents.remove_with_filename_str("tokenizer.json")
///                 .ok_or_else(|| anyhow::anyhow!("missing tokenizer.json"))?.1;
///             let weights = contents.remove_with_filename_str("model.safetensors")
///                 .ok_or_else(|| anyhow::anyhow!("missing model.safetensors"))?.1;
///             Ok(Self { tokenizer, weights })
///         })
///     }
/// }
/// ```
///
/// See also: [`Cache::try_create`], [`CacheClaim`], [`CacheContents`].
pub trait TryFromCache: Sized + MaybeSend {
    /// Declare the set of files needed to construct `Self`.
    ///
    /// The returned future resolves to a list of logical entries (`dirname`/`filename`)
    /// that the caller will fetch and place into [`CacheContents`]. Return an error
    /// if the request is invalid (e.g., unknown key) or if computing the list fails.
    fn claim_files<'a>(
        cache: Cache,
        key: impl AsRef<str>,
        ctx: &'a mut std::collections::HashMap<String, crate::value::Value>,
    ) -> BoxFuture<'a, anyhow::Result<CacheClaim>>;

    /// Build `Self` from the previously fetched files.
    ///
    /// Implementations should verify that all required entries are present and valid,
    /// and return a descriptive `Err(String)` on failure.
    fn try_from_contents<'a>(
        contents: CacheContents,
        ctx: &'a std::collections::HashMap<String, crate::value::Value>,
    ) -> BoxFuture<'a, anyhow::Result<Self>>;
}

/// Infallible variant of [`TryFromCache`].
///
/// Choose `FromCache` when both steps are guaranteed not to fail:
/// - `claim_files` deterministically returns the needed list,
/// - `try_from_contents` cannot fail (e.g., inputs are known-good).
///
/// This trait follows the same pipeline as `TryFromCache`:
/// `claim_files` → (download & aggregate into [`CacheContents`]) → `try_from_contents`.
pub trait FromCache: Sized + MaybeSend {
    /// Declare the set of files needed to construct `Self`.
    ///
    /// This method must not fail; return a complete list of required entries.
    fn claim_files(cache: Cache, key: impl AsRef<str>)
    -> Pin<Box<dyn Future<Output = CacheClaim>>>;

    /// Build `Self` from the previously fetched files.
    ///
    /// Implementations may assume `contents` contain all required entries.
    fn try_from_contents(contents: CacheContents) -> BoxFuture<'static, Self>;
}
