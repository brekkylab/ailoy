#[cfg(target_arch = "wasm32")]
mod wasm {
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};
    use std::task::{Context, Poll};

    pub struct AsyncRwLock<T> {
        inner: RwLock<T>,
    }

    #[cfg(target_arch = "wasm32")]
    impl<T> AsyncRwLock<T> {
        pub fn new(value: T) -> Self {
            Self {
                inner: RwLock::new(value),
            }
        }

        pub fn try_read(&self) -> Option<RwLockReadGuard<'_, T>> {
            self.inner.try_read().ok()
        }

        pub fn try_write(&self) -> Option<RwLockWriteGuard<'_, T>> {
            self.inner.try_write().ok()
        }

        pub fn blocking_read(&self) -> RwLockReadGuard<'_, T> {
            self.inner.read().unwrap()
        }

        pub fn blocking_write(&self) -> RwLockWriteGuard<'_, T> {
            self.inner.write().unwrap()
        }

        pub fn read(&self) -> impl Future<Output = RwLockReadGuard<'_, T>> {
            DummyReadAwaitable::<T> {
                inner: Some(self.blocking_read()),
            }
        }

        pub fn write(&self) -> impl Future<Output = RwLockWriteGuard<'_, T>> {
            DummyWriteAwaitable::<T> {
                inner: Some(self.blocking_write()),
            }
        }
    }

    struct DummyReadAwaitable<'a, T> {
        inner: Option<RwLockReadGuard<'a, T>>,
    }

    impl<'a, T> Future for DummyReadAwaitable<'a, T> {
        type Output = RwLockReadGuard<'a, T>;

        fn poll(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
            match self.inner.take() {
                Some(inner) => Poll::Ready(inner),
                None => panic!(),
            }
        }
    }

    struct DummyWriteAwaitable<'a, T> {
        inner: Option<RwLockWriteGuard<'a, T>>,
    }

    impl<'a, T> Future for DummyWriteAwaitable<'a, T> {
        type Output = RwLockWriteGuard<'a, T>;

        fn poll(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
            match self.inner.take() {
                Some(inner) => Poll::Ready(inner),
                None => panic!(),
            }
        }
    }
}

#[cfg(target_arch = "wasm32")]
pub type RwLock<T> = wasm::AsyncRwLock<T>;

#[cfg(not(target_arch = "wasm32"))]
pub type RwLock<T> = tokio::sync::RwLock<T>;
