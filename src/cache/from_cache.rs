use std::pin::Pin;

use crate::{
    cache::{Cache, CacheClaim, CacheContents},
    utils::{BoxFuture, MaybeSend},
};

/// Build a value by fetching the files it needs from the Ailoy [`Cache`].
///
/// # How it works
/// The construction pipeline is:
/// 1. **`claim_files(cache, key)`** → declare which logical files (`CacheEntry`s)
///    are required to build `Self`. Optionally, attach a `ctx` (boxed `Any`)
///    containing extra metadata needed for reconstruction.
/// 2. The caller (typically [`Cache::try_create`]) **loads (or downloads)** those files
///    and **aggregates** them into a [`CacheContents`] (handling local/remote logic).
/// 3. **`try_from_contents(contents)`** → asynchronously parse/assemble `Self` from
///    the provided bytes in `contents`.
///
/// # Contracts (rules you must follow)
/// - **`claim_files` should be lightweight**: its main job is to declare which
///   logical files are required. It should not perform heavy work such as
///   reading file contents or changing state. At most, it may inspect `cache`
///   (e.g., to check a root path or remote URL).
/// - **File names must match the manifest**: the logical names returned from
///   `claim_files` must exist in the remote directory manifest (`dirname`/`filename`).
/// - **`try_from_contents` must validate and consume files**: it takes ownership
///   of `CacheContents`, extracts the required entries, and removes them as it goes.
///   Since LLM models can include very large files, avoid copying data unnecessarily,
///   as this may lead to out-of-memory issues.
/// - **If you use `ctx`**: both `claim_files` and `try_from_contents` must agree
///   on how to interpret it. `ctx` is just a generic box (`Any`) that carries
///   extra information between the two steps. If no extra info is needed, you
///   can leave it empty.
///
/// # Example
/// ```rust,ignore
/// struct MyModel { tokenizer: Vec<u8>, weights: Vec<u8> }
///
/// impl TryFromCache for MyModel {
///     fn claim_files(cache: Cache, key: impl AsRef<str>)
///     -> BoxFuture<'static, anyhow::Result<CacheClaim>> {
///         let k = key.as_ref().to_owned();
///         Box::pin(async move {
///             Ok(CacheClaim::with_ctx(
///                 vec![
///                     CacheEntry::new(&k, "tokenizer.json"),
///                     CacheEntry::new(&k, "model.safetensors"),
///                 ],
///                 Box::new(()), // empty ctx in this simple case
///             ))
///         })
///     }
///
///     fn try_from_contents(contents: CacheContents)
///     -> BoxFuture<'static, anyhow::Result<Self>> {
///         Box::pin(async move {
///             let tokenizer = contents.remove_with_filename_str("tokenizer.json")
///                 .ok_or("missing tokenizer.json")?.1;
///             let weights = contents.remove_with_filename_str("model.safetensors")
///                 .ok_or("missing model.safetensors")?.1;
///             Ok(MyModel { tokenizer, weights })
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
    fn claim_files(
        cache: Cache,
        key: impl AsRef<str>,
    ) -> BoxFuture<'static, anyhow::Result<CacheClaim>>;

    /// Build `Self` from the previously fetched files.
    ///
    /// Implementations should verify that all required entries are present and valid,
    /// and return a descriptive `Err(String)` on failure.
    fn try_from_contents(contents: CacheContents) -> BoxFuture<'static, anyhow::Result<Self>>;
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
