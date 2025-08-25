#[cfg(test)]
#[cfg(not(target_arch = "wasm32"))]
#[macro_export]
macro_rules! multi_platform_test {
    ($(#[$meta:meta])* $vis:vis async fn $name:ident($($args:tt)*) $body:block) => {
        $(#[$meta])*
        #[tokio::test]
        $vis async fn $name($($args)*) {
            let _ = env_logger::builder().is_test(true).try_init();
            $body
        }
    };
}

#[cfg(test)]
#[cfg(target_arch = "wasm32")]
#[macro_export]
macro_rules! multi_platform_test {
    ($(#[$meta:meta])* $vis:vis async fn $name:ident($($args:tt)*) $body:block) => {
        $(#[$meta])*
        #[wasm_bindgen_test::wasm_bindgen_test]
        $vis async fn $name($($args)*) $body
    };
}
