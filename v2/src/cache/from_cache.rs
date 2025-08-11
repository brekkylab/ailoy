use std::pin::Pin;

use crate::cache::{Cache, CacheContents, CacheEntry};

/// Build a value by fetching the files it needs from the Ailoy [`Cache`].
///
/// # How it works
/// The construction pipeline is:
/// 1. **`claim_files(cache, key)`** → declare which logical files (`CacheEntry`s)
///    are required to build `Self`.
/// 2. The caller (typically [`Cache::try_create`]) **loads(or downloads)** those files
///    and **aggregates** them into a [`CacheContents`] (handling local/remote logic).
/// 3. **`try_from_contents(contents)`** → parse/assemble `Self` from the provided
///    bytes in `contents`.
///
/// Use `TryFromCache` when either discovering the file list or constructing the type
/// can fail (e.g., remote lookup, missing files, decode errors). If both steps are
/// infallible in your case, see [`FromCache`] for a simpler variant.
///
/// Implementations should treat `key` as a free-form selector (e.g., a model ID).
/// `claim_files` should be **pure** (idempotent; no writes) and may inspect `cache`
/// (e.g., for root or remote URL) to compute paths.
///
/// # Contracts
/// - `claim_files` must return logical filenames (by `dirname`/`filename`) that will
///   be present in the remote directory manifest(s).
/// - `try_from_contents` should *only* read from `contents` and return a meaningful
///   error if any required entry is absent or malformed.
/// - Implementations are free to consume entries from `contents` (e.g., via
///   remove-like APIs) to detect missing files.
///
/// # Example
/// ```rust,ignore
/// struct MyModel { tokenizer: Vec<u8>, weights: Vec<u8> }
///
/// impl TryFromCache for MyModel {
///     fn claim_files(
///         _cache: Cache,
///         key: impl AsRef<str>,
///     ) -> Pin<Box<dyn Future<Output = Result<Vec<CacheEntry>, String>>>> {
///         let k = key.as_ref().to_owned();
///         Box::pin(async move {
///             Ok(vec![
///                 CacheEntry::new(&format!("{k}"), "tokenizer.json"),
///                 CacheEntry::new(&format!("{k}"), "model.safetensors"),
///             ])
///         })
///     }
///
///     fn try_from_contents(contents: &mut CacheContents) -> Result<Self, String> {
///         let tokenizer = contents.remove_with_filename_str("tokenizer.json")
///             .ok_or("missing tokenizer.json")?.1;
///         let weights = contents.remove_with_filename_str("model.safetensors")
///             .ok_or("missing model.safetensors")?.1;
///         Ok(MyModel { tokenizer, weights })
///     }
/// }
/// ```
///
/// See also: [`Cache::try_create`], [`CacheEntry`], [`CacheContents`].
pub trait TryFromCache {
    /// Declare the set of files needed to construct `Self`.
    ///
    /// The returned future resolves to a list of logical entries (`dirname`/`filename`)
    /// that the caller will fetch and place into [`CacheContents`]. Return an error
    /// if the request is invalid (e.g., unknown key) or if computing the list fails.
    fn claim_files(
        cache: Cache,
        key: impl AsRef<str>,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<CacheEntry>, String>>>>;

    /// Build `Self` from the previously fetched files.
    ///
    /// Implementations should verify that all required entries are present and valid,
    /// and return a descriptive `Err(String)` on failure.
    fn try_from_contents(contents: &mut CacheContents) -> Result<Self, String>
    where
        Self: Sized;
}

/// Infallible variant of [`TryFromCache`].
///
/// Choose `FromCache` when both steps are guaranteed not to fail:
/// - `claim_files` deterministically returns the needed list,
/// - `try_from_contents` cannot fail (e.g., inputs are known-good).
///
/// This trait follows the same pipeline as `TryFromCache`:
/// `claim_files` → (download & aggregate into [`CacheContents`]) → `try_from_contents`.
pub trait FromCache {
    /// Declare the set of files needed to construct `Self`.
    ///
    /// This method must not fail; return a complete list of required entries.
    fn claim_files(
        cache: Cache,
        key: impl AsRef<str>,
    ) -> Pin<Box<dyn Future<Output = Vec<CacheEntry>>>>;

    /// Build `Self` from the previously fetched files.
    ///
    /// Implementations may assume `contents` contain all required entries.
    fn try_from_contents(contents: &mut CacheContents) -> Self
    where
        Self: Sized;
}
