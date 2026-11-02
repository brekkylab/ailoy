use std::io::Cursor;

use image::{ImageFormat, imageops::FilterType};

use crate::message::{Message, Part, PartImage};

/// Longest edge an image keeps, in pixels: the largest any Claude model takes natively.
/// Past it the API downscales anyway, and in a request of over twenty images refuses any
/// edge over 3000. Fitting before the first send keeps the history, resent every call,
/// within what the API takes.
pub(crate) const MAX_IMAGE_EDGE: u32 = 2576;

/// Shrink every embedded image of `message` to fit [`MAX_IMAGE_EDGE`], keeping its shape
/// and format. An image that fits, is reached by URL, or cannot be read is left as it is.
pub(crate) fn fit_images(message: &mut Message) {
    for part in &mut message.contents {
        if let Part::Image {
            image: PartImage::Embedded { mime_type, data },
        } = part
            && let Some((format, fitted)) = fit(data.as_ref())
        {
            *mime_type = format.to_mime_type().to_owned();
            *data = fitted.into();
        }
    }
}

/// `data` re-encoded with its longest edge at [`MAX_IMAGE_EDGE`]; `None` when it fits
/// already or is not an image this build can decode and encode.
fn fit(data: &[u8]) -> Option<(ImageFormat, Vec<u8>)> {
    let format = image::guess_format(data).ok()?;
    if !format.writing_enabled() {
        return None;
    }
    let picture = image::load_from_memory_with_format(data, format).ok()?;
    let (width, height) = (picture.width(), picture.height());
    if width.max(height) <= MAX_IMAGE_EDGE {
        return None;
    }
    // `resize` keeps the aspect ratio within the box, so the longer edge lands on the limit.
    let fitted = picture.resize(MAX_IMAGE_EDGE, MAX_IMAGE_EDGE, FilterType::Triangle);
    let mut out = Vec::new();
    fitted.write_to(&mut Cursor::new(&mut out), format).ok()?;
    Some((format, out))
}

#[cfg(test)]
mod tests {
    use image::{DynamicImage, RgbaImage};

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
            let Part::Image {
                image: PartImage::Embedded { mime_type, .. },
            } = &message.contents[0]
            else {
                panic!("{row}: not an embedded image");
            };
            assert_eq!(mime_type, mime, "{row}");
        }
    }

    #[test]
    fn test_the_longer_edge_counts_whichever_it_is() {
        let data = encoded(100, 2577, ImageFormat::Png);
        let part = Part::image_embedded("image/png", Bytes::from(data)).unwrap();
        let mut message = Message::new(Role::User).with_contents([part]);
        fit_images(&mut message);
        assert_eq!(dimensions(&message.contents[0]), (100, 2576));
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
