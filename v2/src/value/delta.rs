pub trait Delta: Default {
    type Item;

    /// similar to add operator overloading
    ///
    /// Raises error when enum is different between self & other
    fn aggregate(self, other: Self) -> Result<Self, ()>;

    fn finish(self) -> Result<Self::Item, String>;
}
