pub trait Delta: Default {
    type Item;
    type Err;

    /// Merge `other` into `self`, like `+`.
    ///
    /// Errors when `self` and `other` are incompatible variants.
    fn accumulate(self, other: Self) -> Result<Self, Self::Err>;

    fn finish(self) -> Result<Self::Item, Self::Err>;
}
