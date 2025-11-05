#[cfg(target_arch = "wasm32")]
pub async fn sleep(millis: i32) {
    gloo_timers::future::sleep(std::time::Duration::from_millis(millis as u64)).await;
}

#[cfg(not(target_arch = "wasm32"))]
pub async fn sleep(millis: i32) {
    tokio::time::sleep(std::time::Duration::from_millis(millis as u64)).await;
}
