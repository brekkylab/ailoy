fn main() -> anyhow::Result<()> {
    #[cfg(feature = "python")]
    {
        // `stub_info` is a function defined by `define_stub_info_gatherer!` macro.
        let stub = ailoy::ffi::py::stub_info()?;
        stub.generate()?;
    }
    Ok(())
}
