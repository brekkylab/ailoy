pub trait Delta: Default {
    type Item;
    type Err;

    /// similar to add operator overloading
    ///
    /// Raises error when enum is different between self & other
    fn accumulate(self, other: Self) -> Result<Self, Self::Err>;

    fn finish(self) -> Result<Self::Item, Self::Err>;
}
