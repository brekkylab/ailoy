use std::borrow::Cow;
use unicode_segmentation::UnicodeSegmentation;

pub trait Ellipsis {
    type Output;

    /// Truncate to a length of `len` extended grapheme clusters and place the given
    /// ellipsis string at the end when truncating.
    ///
    /// Truncating to a length of 0 will yield the empty element without an
    /// attached ellipsis.
    fn truncate_ellipsis_with(&self, len: usize, ellipsis: &str) -> Self::Output;

    /// Truncate to a length of `len` extended grapheme clusters and add `...` at
    /// the end of the string when truncating.
    ///
    /// Truncating to a length of 0 will yield the empty element without an
    /// attached ellipsis.
    fn truncate_ellipsis(&self, len: usize) -> Self::Output {
        self.truncate_ellipsis_with(len, "...")
    }
}

impl<'a> Ellipsis for &'a str {
    type Output = Cow<'a, str>;

    fn truncate_ellipsis_with(&self, len: usize, ellipsis: &str) -> Self::Output {
        if self.graphemes(true).count() <= len {
            return Cow::Borrowed(self);
        } else if len == 0 {
            return Cow::Borrowed("");
        }

        let result = self
            .graphemes(true)
            .take(len)
            .chain(ellipsis.graphemes(true))
            .collect();
        Cow::Owned(result)
    }
}

impl Ellipsis for String {
    type Output = String;

    fn truncate_ellipsis_with(&self, len: usize, ellipsis: &str) -> Self::Output {
        if self.graphemes(true).count() <= len {
            return self.into();
        } else if len == 0 {
            return "".into();
        }

        let result = self
            .graphemes(true)
            .take(len)
            .chain(ellipsis.graphemes(true))
            .collect();
        result
    }
}
