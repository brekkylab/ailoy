use std::borrow::Cow;
use unicode_segmentation::UnicodeSegmentation;

pub trait Ellipse {
    type Output;

    /// Truncate to a length of `len` extended grapheme clusters and place the given
    /// ellipse string at the end when truncating.
    ///
    /// Truncating to a length of 0 will yield the empty element without an
    /// attached ellipsis.
    fn truncate_ellipse_with(&self, len: usize, ellipse: &str) -> Self::Output;

    /// Truncate to a length of `len` extended grapheme clusters and add `...` at
    /// the end of the string when truncating.
    ///
    /// Truncating to a length of 0 will yield the empty element without an
    /// attached ellipsis.
    fn truncate_ellipse(&self, len: usize) -> Self::Output {
        self.truncate_ellipse_with(len, "...")
    }
}

impl<'a> Ellipse for &'a str {
    type Output = Cow<'a, str>;

    fn truncate_ellipse_with(&self, len: usize, ellipse: &str) -> Self::Output {
        if self.graphemes(true).count() <= len {
            return Cow::Borrowed(self);
        } else if len == 0 {
            return Cow::Borrowed("");
        }

        let result = self
            .graphemes(true)
            .take(len)
            .chain(ellipse.graphemes(true))
            .collect();
        Cow::Owned(result)
    }
}

impl Ellipse for String {
    type Output = String;

    fn truncate_ellipse_with(&self, len: usize, ellipse: &str) -> Self::Output {
        if self.graphemes(true).count() <= len {
            return self.into();
        } else if len == 0 {
            return "".into();
        }

        let result = self
            .graphemes(true)
            .take(len)
            .chain(ellipse.graphemes(true))
            .collect();
        result
    }
}
