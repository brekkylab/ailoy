/// Suppress panic hook output for the duration of the enclosing scope.
///
/// The panic hook fires before `catch_unwind`, so intentional test panics print spurious
/// "panicked at ..." lines. The original hook is restored on drop, including on
/// assertion failure.
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
