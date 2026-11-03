use std::io::Cursor;

use image::{
    DynamicImage, ImageDecoder, ImageEncoder, ImageFormat, ImageReader,
    codecs::{
        jpeg::JpegEncoder,
        png::{PngDecoder, PngEncoder},
    },
    imageops::FilterType,
};

use crate::message::{Message, Part, PartImage};

/// Longest edge an image keeps, in pixels: the largest any Claude model takes natively.
/// Past it the API downscales anyway, and in a request of over twenty images refuses any
/// edge over 3000. Fitting before the first send keeps the history, resent every call,
/// within what the API takes.
pub(crate) const MAX_IMAGE_EDGE: u32 = 2576;

/// JPEG quality a fitted image is written at, high enough to keep small text legible.
const JPEG_QUALITY: u8 = 90;

/// Largest PNG an opaque lossless WebP is fitted into before it goes as JPEG instead: 5 MB
/// in base64, Bedrock's limit for one image.
const MAX_PNG_FROM_WEBP_BYTES: usize = 3_750_000;

/// Shrink every embedded image of `message` to fit [`MAX_IMAGE_EDGE`], keeping its shape.
/// An image that fits, is reached by URL, cannot be read, or is animated is left as it is:
/// what becomes of an animation's frames is the model's API to decide. So is one whose
/// decoder panics, so that a malformed image never takes the turn down.
pub(crate) fn fit_images(message: &mut Message) {
    for part in &mut message.contents {
        if let Part::Image {
            image: PartImage::Embedded { mime_type, data },
        } = part
            && let Ok(Some((format, fitted))) = std::panic::catch_unwind(|| fit(data.as_ref()))
        {
            *mime_type = format.to_mime_type().to_owned();
            *data = fitted.into();
        }
    }
}

/// [`fit_images`] on a blocking thread, since decoding and encoding a large image takes
/// a runtime thread for up to a few hundred milliseconds. A message with no embedded image
/// stays where it is, and one whose thread never ran, as when the runtime shuts down, comes
/// back unfitted.
pub(crate) async fn fit_images_off_runtime(message: Message) -> Message {
    let embedded = message.contents.iter().any(|part| {
        matches!(
            part,
            Part::Image {
                image: PartImage::Embedded { .. }
            }
        )
    });
    if !embedded {
        return message;
    }
    let mut fitted = message.clone();
    tokio::task::spawn_blocking(move || {
        fit_images(&mut fitted);
        fitted
    })
    .await
    .unwrap_or(message)
}

/// `data` re-encoded with its longest edge at [`MAX_IMAGE_EDGE`], turned upright by its EXIF
/// orientation and keeping its color profile; `None` when it fits already or is not an
/// image this build can decode and encode.
///
/// The `image` crate writes WebP lossless only, which makes a lossy photo several times
/// larger, so a WebP comes back as PNG when it has transparent pixels or was lossless
/// (a screenshot or a drawing, whose text JPEG would blur), and as JPEG otherwise.
fn fit(data: &[u8]) -> Option<(ImageFormat, Vec<u8>)> {
    let format = image::guess_format(data).ok()?;
    if !format.writing_enabled() {
        return None;
    }
    let mut decoder = ImageReader::with_format(Cursor::new(data), format)
        .into_decoder()
        .ok()?;
    let (width, height) = decoder.dimensions();
    if width.max(height) <= MAX_IMAGE_EDGE || is_animated(data, format) {
        return None;
    }
    let orientation = decoder.orientation().ok()?;
    let icc = decoder.icc_profile().ok().flatten();
    let mut picture = DynamicImage::from_decoder(decoder).ok()?;
    picture.apply_orientation(orientation);
    // `resize` keeps the aspect ratio within the box, so the longer edge lands on the limit.
    let fitted = picture.resize(MAX_IMAGE_EDGE, MAX_IMAGE_EDGE, FilterType::Triangle);

    if format != ImageFormat::WebP {
        return Some((format, encode(&fitted, format, icc)?));
    }
    // Judged before the resize, whose filter may leave an opaque edge just short of opaque.
    if has_transparency(&picture) {
        return Some((ImageFormat::Png, encode(&fitted, ImageFormat::Png, icc)?));
    }
    let opaque = DynamicImage::ImageRgb8(fitted.to_rgb8());
    if is_lossless_webp(data)
        && let Some(png) = encode(&opaque, ImageFormat::Png, icc.clone())
        && png.len() <= MAX_PNG_FROM_WEBP_BYTES
    {
        return Some((ImageFormat::Png, png));
    }
    Some((ImageFormat::Jpeg, encode(&opaque, ImageFormat::Jpeg, icc)?))
}

/// `picture` written as `format`, carrying `icc` where the format holds one.
fn encode(picture: &DynamicImage, format: ImageFormat, icc: Option<Vec<u8>>) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    match format {
        ImageFormat::Jpeg => {
            let mut encoder = JpegEncoder::new_with_quality(&mut out, JPEG_QUALITY);
            if let Some(icc) = icc {
                let _ = encoder.set_icc_profile(icc);
            }
            picture.write_with_encoder(encoder).ok()?;
        }
        ImageFormat::Png => {
            let mut encoder = PngEncoder::new(&mut out);
            if let Some(icc) = icc {
                let _ = encoder.set_icc_profile(icc);
            }
            picture.write_with_encoder(encoder).ok()?;
        }
        format => picture.write_to(&mut Cursor::new(&mut out), format).ok()?,
    }
    Some(out)
}

/// Whether a pixel of `picture` is less than opaque; an alpha channel alone, common in
/// opaque WebP photos, does not count.
fn has_transparency(picture: &DynamicImage) -> bool {
    match picture {
        DynamicImage::ImageRgba8(p) => p.pixels().any(|px| px[3] < u8::MAX),
        DynamicImage::ImageLumaA8(p) => p.pixels().any(|px| px[1] < u8::MAX),
        DynamicImage::ImageRgba16(p) => p.pixels().any(|px| px[3] < u16::MAX),
        DynamicImage::ImageLumaA16(p) => p.pixels().any(|px| px[1] < u16::MAX),
        DynamicImage::ImageRgba32F(p) => p.pixels().any(|px| px[3] < 1.0),
        picture => picture.color().has_alpha(),
    }
}

/// Whether `data` is an animated WebP or an APNG. A GIF is never fitted, since this build
/// does not decode it.
fn is_animated(data: &[u8], format: ImageFormat) -> bool {
    match format {
        ImageFormat::WebP => webp_chunks(data).any(|chunk| chunk == b"ANIM"),
        ImageFormat::Png => {
            PngDecoder::new(Cursor::new(data)).is_ok_and(|png| png.is_apng().unwrap_or(false))
        }
        _ => false,
    }
}

/// Whether the WebP `data` holds a lossless (`VP8L`) bitstream rather than a lossy (`VP8 `)
/// one.
fn is_lossless_webp(data: &[u8]) -> bool {
    webp_chunks(data)
        .find(|chunk| *chunk == b"VP8L" || *chunk == b"VP8 ")
        .is_some_and(|chunk| chunk == b"VP8L")
}

/// The ids of the RIFF chunks of the WebP `data`, in order.
fn webp_chunks(data: &[u8]) -> impl Iterator<Item = &[u8]> {
    let mut at = 12;
    std::iter::from_fn(move || {
        let head = data.get(at..at + 8)?;
        let size = u32::from_le_bytes([head[4], head[5], head[6], head[7]]) as usize;
        at = at.saturating_add(8 + size + size % 2);
        Some(&head[..4])
    })
}

#[cfg(test)]
mod tests {
    use image::{RgbImage, RgbaImage};

    use super::*;
    use crate::datatype::Bytes;
    use crate::message::Role;

    fn encoded(width: u32, height: u32, format: ImageFormat) -> Vec<u8> {
        let mut out = Vec::new();
        DynamicImage::ImageRgba8(RgbaImage::new(width, height))
            .write_to(&mut Cursor::new(&mut out), format)
            .unwrap();
        out
    }

    fn dimensions(part: &Part) -> (u32, u32) {
        let Part::Image {
            image: PartImage::Embedded { data, .. },
        } = part
        else {
            panic!("not an embedded image");
        };
        let picture = image::load_from_memory(data.as_ref()).unwrap();
        (picture.width(), picture.height())
    }

    #[test]
    fn test_an_image_over_the_edge_is_shrunk_to_it_in_its_own_format() {
        for (row, format, mime) in [
            ("png", ImageFormat::Png, "image/png"),
            ("jpeg", ImageFormat::Jpeg, "image/jpeg"),
        ] {
            let data = encoded(3000, 100, format);
            let part = Part::image_embedded(mime, Bytes::from(data)).unwrap();
            let mut message = Message::new(Role::Tool).with_contents([part]);
            fit_images(&mut message);

            assert_eq!(dimensions(&message.contents[0]), (2576, 86), "{row}");
            assert_eq!(mime_of(&message.contents[0]), mime, "{row}");
        }
    }

    fn mime_of(part: &Part) -> &str {
        let Part::Image {
            image: PartImage::Embedded { mime_type, .. },
        } = part
        else {
            panic!("not an embedded image");
        };
        mime_type
    }

    #[test]
    fn test_a_jpeg_is_turned_upright_by_its_exif_orientation() {
        // Stored 3000 wide and 100 high with Orientation 6, so shown a quarter turn
        // clockwise: 100 wide and 3000 high.
        let exif = [
            0x49, 0x49, 0x2A, 0x00, 0x08, 0x00, 0x00, 0x00, // little-endian TIFF, IFD at 8
            0x01, 0x00, // one entry
            0x12, 0x01, 0x03, 0x00, // Orientation, a SHORT
            0x01, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00, // one value: 6
            0x00, 0x00, 0x00, 0x00, // no next IFD
        ];
        let mut data = Vec::new();
        let mut encoder = JpegEncoder::new(&mut data);
        encoder.set_exif_metadata(exif.to_vec()).unwrap();
        DynamicImage::ImageRgb8(RgbImage::new(3000, 100))
            .write_with_encoder(encoder)
            .unwrap();
        let part = Part::image_embedded("image/jpeg", Bytes::from(data)).unwrap();
        let mut message = Message::new(Role::User).with_contents([part]);
        fit_images(&mut message);
        assert_eq!(dimensions(&message.contents[0]), (86, 2576));
    }

    #[test]
    fn test_the_color_profile_is_kept() {
        let icc = b"a color profile".to_vec();
        let mut data = Vec::new();
        let mut encoder = PngEncoder::new(&mut data);
        encoder.set_icc_profile(icc.clone()).unwrap();
        DynamicImage::ImageRgb8(RgbImage::new(3000, 100))
            .write_with_encoder(encoder)
            .unwrap();
        let part = Part::image_embedded("image/png", Bytes::from(data)).unwrap();
        let mut message = Message::new(Role::User).with_contents([part]);
        fit_images(&mut message);

        let Part::Image {
            image: PartImage::Embedded { data, .. },
        } = &message.contents[0]
        else {
            panic!("not an embedded image");
        };
        let mut decoder = ImageReader::new(Cursor::new(data.as_ref()))
            .with_guessed_format()
            .unwrap()
            .into_decoder()
            .unwrap();
        assert_eq!(decoder.dimensions(), (2576, 86));
        assert_eq!(decoder.icc_profile().unwrap(), Some(icc));
    }

    fn webp(picture: DynamicImage) -> Part {
        let mut data = Vec::new();
        picture
            .write_to(&mut Cursor::new(&mut data), ImageFormat::WebP)
            .unwrap();
        Part::image_embedded("image/webp", Bytes::from(data)).unwrap()
    }

    /// The `image` crate writes WebP lossless only, so each of these is lossless.
    #[test]
    fn test_a_lossless_webp_comes_back_as_png() {
        for (row, picture) in [
            ("opaque", DynamicImage::ImageRgb8(RgbImage::new(3000, 100))),
            (
                "an alpha channel, all opaque",
                DynamicImage::ImageRgba8(RgbaImage::from_pixel(3000, 100, [0, 0, 0, 255].into())),
            ),
            (
                "transparent",
                DynamicImage::ImageRgba8(RgbaImage::new(3000, 100)),
            ),
        ] {
            let mut message = Message::new(Role::User).with_contents([webp(picture)]);
            fit_images(&mut message);
            assert_eq!(mime_of(&message.contents[0]), "image/png", "{row}");
            assert_eq!(dimensions(&message.contents[0]), (2576, 86), "{row}");
        }
    }

    #[test]
    fn test_an_opaque_webp_too_large_as_png_comes_back_as_jpeg() {
        // Noise compresses poorly: about 8 MB as a 2576-wide PNG.
        let mut seed = 1u32;
        let noise = RgbImage::from_fn(2600, 1000, |_, _| {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let [r, g, b, _] = seed.to_le_bytes();
            [r, g, b].into()
        });
        let mut message =
            Message::new(Role::User).with_contents([webp(DynamicImage::ImageRgb8(noise))]);
        fit_images(&mut message);
        assert_eq!(mime_of(&message.contents[0]), "image/jpeg");
    }

    #[test]
    fn test_what_fits_or_cannot_be_read_is_left_alone() {
        let fits = Part::image_embedded(
            "image/png",
            Bytes::from(encoded(MAX_IMAGE_EDGE, 10, ImageFormat::Png)),
        )
        .unwrap();
        let junk = Part::image_embedded("image/png", Bytes::from(vec![0u8; 4])).unwrap();
        let url = Part::image_url("https://example.com/a.png".into()).unwrap();
        let text = Part::text("the page");
        let parts = [fits, junk, url, text];
        let mut message = Message::new(Role::Tool).with_contents(parts.clone());
        fit_images(&mut message);
        assert_eq!(message.contents, parts);
    }
}
