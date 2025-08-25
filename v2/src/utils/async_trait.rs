#[cfg(not(target_arch = "wasm32"))]
#[macro_export]
macro_rules! async_trait {
    ($item:item) => {
        #[async_trait::async_trait]
        $item
    };
}

#[cfg(target_arch = "wasm32")]
#[macro_export]
macro_rules! async_trait {
    ($item:item) => {
        #[async_trait::async_trait(?Send)]
        $item
    };
}
