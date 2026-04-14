/// Suppress panic hook output for the duration of the enclosing scope.
///
/// Rust's panic hook fires before `catch_unwind` catches a panic, so
/// intentional panics in tests produce spurious "panicked at ..." lines.
/// This macro replaces the hook with a no-op and restores the original
/// hook when the binding is dropped (including on assertion failure).
#[macro_export]
macro_rules! suppress_panics {
    () => {
        let _panic_suppressor = {
            struct Suppressor(Option<Box<dyn Fn(&std::panic::PanicHookInfo<'_>) + Send + Sync>>);
            impl Drop for Suppressor {
                fn drop(&mut self) {
                    if let Some(hook) = self.0.take() {
                        std::panic::set_hook(hook);
                    }
                }
            }
            let prev = std::panic::take_hook();
            std::panic::set_hook(Box::new(|_| {}));
            Suppressor(Some(prev))
        };
    };
}
