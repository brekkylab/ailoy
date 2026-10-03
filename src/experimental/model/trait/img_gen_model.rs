use futures::future::BoxFuture;

use crate::message::PartImage;

/// A model that generates images from a text prompt (gpt-image, Imagen, FLUX). Size, count,
/// quality and the like differ by vendor, so each implementor carries its own options.
pub trait ImgGenModelInference: Send + Sync {
    /// Generates images for `prompt`. Non-empty `images` are references to edit or follow,
    /// for models that take them. Results are embedded or by URL, as the vendor returns them.
    fn infer<'a>(
        &'a self,
        prompt: &'a str,
        images: &'a [PartImage],
    ) -> BoxFuture<'a, anyhow::Result<Vec<PartImage>>>;
}
