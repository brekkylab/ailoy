#[cfg(not(target_arch = "wasm32"))]
mod sync {
    use core::{future::Future, pin::Pin};

    use futures::stream::Stream;

    /// Reexports of the actual marker traits from core.
    pub use core::marker::{Send as MaybeSend, Sync as MaybeSync};

    pub type BoxFuture<'a, T> = Pin<alloc::boxed::Box<dyn Future<Output = T> + Send + 'a>>;

    pub type BoxStream<'a, T> = Pin<alloc::boxed::Box<dyn Stream<Item = T> + Send + 'a>>;

    /// A pointer type which can be safely shared between threads
    /// when "sync" feature is enabled.\
    /// A pointer type which can be shared, but only within single thread
    /// where it was created when "sync" feature is not enabled.
    ///
    /// # Example
    ///
    /// ```
    /// # use {maybe_sync::{MaybeSend, Rc}, std::fmt::Debug};
    ///
    /// fn maybe_sends<T: MaybeSend + Debug + 'static>(val: T) {
    ///   #[cfg(feature = "sync")]
    ///   {
    ///     // If this code is compiled then `MaybeSend` is alias to `std::marker::Send`.
    ///     std::thread::spawn(move || { println!("{:?}", val) });
    ///   }
    /// }
    ///
    /// // Unlike `std::rc::Rc` this `maybe_sync::Rc` always satisfies `MaybeSend` bound.
    /// maybe_sends(Rc::new(42));
    /// ```
    pub type Rc<T> = alloc::sync::Arc<T>;

    /// Mutex implementation to use in conjunction with `MaybeSync` bound.
    ///
    /// A type alias to `parking_lot::Mutex` when "sync" feature is enabled.\
    /// A wrapper type around `std::cell::RefCell` when "sync" feature is not enabled.
    ///
    /// # Example
    ///
    /// ```
    /// # use {maybe_sync::{MaybeSend, Mutex}, std::{fmt::Debug, sync::Arc}};
    ///
    /// fn maybe_sends<T: MaybeSend + Debug + 'static>(val: Arc<Mutex<T>>) {
    ///   #[cfg(feature = "sync")]
    ///   {
    ///     // If this code is compiled then `MaybeSend` is alias to `std::marker::Send`,
    ///     // and `Mutex` is `parking_lot::Mutex`.
    ///     std::thread::spawn(move || { println!("{:?}", *val.lock()) });
    ///   }
    /// }
    ///
    /// // `maybe_sync::Mutex<T>` would always satisfy `MaybeSync` and `MaybeSend`
    /// // bounds when `T: MaybeSend`,
    /// // even if feature "sync" is enabeld.
    /// maybe_sends(Arc::new(Mutex::new(42)));
    /// ```
    pub type Mutex<T> = parking_lot::Mutex<T>;

    /// A boolean type which can be safely shared between threads
    /// when "sync" feature is enabled.\
    /// A boolean type with non-threadsafe interior mutability
    /// when "sync" feature is not enabled.
    ///
    /// This type has the same in-memory representation as a bool.
    pub type AtomicBool = core::sync::atomic::AtomicBool;

    /// A integer type which can be safely shared between threads
    /// when "sync" feature is enabled.\
    /// A integer type with non-threadsafe interior mutability
    /// when "sync" feature is not enabled.
    ///
    /// This type has the same in-memory representation as a i8.
    pub type AtomicI8 = core::sync::atomic::AtomicI8;

    /// A integer type which can be safely shared between threads
    /// when "sync" feature is enabled.\
    /// A integer type with non-threadsafe interior mutability
    /// when "sync" feature is not enabled.
    ///
    /// This type has the same in-memory representation as a i16.
    pub type AtomicI16 = core::sync::atomic::AtomicI16;

    /// A integer type which can be safely shared between threads
    /// when "sync" feature is enabled.\
    /// A integer type with non-threadsafe interior mutability
    /// when "sync" feature is not enabled.
    ///
    /// This type has the same in-memory representation as a i32.
    pub type AtomicI32 = core::sync::atomic::AtomicI32;

    /// A integer type which can be safely shared between threads
    /// when "sync" feature is enabled.\
    /// A integer type with non-threadsafe interior mutability
    /// when "sync" feature is not enabled.
    ///
    /// This type has the same in-memory representation as a isize.
    pub type AtomicIsize = core::sync::atomic::AtomicIsize;

    /// A integer type which can be safely shared between threads
    /// when "sync" feature is enabled.\
    /// A integer type with non-threadsafe interior mutability
    /// when "sync" feature is not enabled.
    ///
    /// This type has the same in-memory representation as a i8.
    pub type AtomicU8 = core::sync::atomic::AtomicU8;

    /// A integer type which can be safely shared between threads
    /// when "sync" feature is enabled.\
    /// A integer type with non-threadsafe interior mutability
    /// when "sync" feature is not enabled.
    ///
    /// This type has the same in-memory representation as a i16.
    pub type AtomicU16 = core::sync::atomic::AtomicU16;

    /// A integer type which can be safely shared between threads
    /// when "sync" feature is enabled.\
    /// A integer type with non-threadsafe interior mutability
    /// when "sync" feature is not enabled.
    ///
    /// This type has the same in-memory representation as a i32.
    pub type AtomicU32 = core::sync::atomic::AtomicU32;

    /// A integer type which can be safely shared between threads
    /// when "sync" feature is enabled.\
    /// A integer type with non-threadsafe interior mutability
    /// when "sync" feature is not enabled.
    ///
    /// This type has the same in-memory representation as a isize.
    pub type AtomicUsize = core::sync::atomic::AtomicUsize;

    /// A raw pointer type which can be safely shared between threads
    /// when "sync" feature is enabled.\
    /// A raw pointer type with non-threadsafe interior mutability
    /// when "sync" feature is not enabled.
    ///
    /// This type has the same in-memory representation as a isize.
    pub type AtomicPtr<T> = core::sync::atomic::AtomicPtr<T>;
}

#[cfg(target_arch = "wasm32")]
mod unsync {
    use core::cell::{RefCell, RefMut};

    use core::{future::Future, pin::Pin};

    use futures::stream::Stream;

    /// Marker trait that can represent nothing if feature "sync" is not enabled.
    /// Or be reexport of `std::marker::Send` if "sync" feature is enabled.
    ///
    /// It is intended to be used as trait bound where `std::marker::Send` bound
    /// is required only when application is compiled for multithreaded environment.\
    /// If "sync" feature is not enabled then this trait bound will *NOT* allow
    /// value to cross thread boundary or be used where sendable value is expected.
    ///
    /// # Examples
    ///
    /// ```
    /// # use {maybe_sync::MaybeSend, std::{fmt::Debug, rc::Rc}};
    ///
    /// fn maybe_sends<T: MaybeSend + Debug + 'static>(val: T) {
    ///   #[cfg(feature = "sync")]
    ///   {
    ///     // If this code is compiled then `MaybeSend` is alias to `std::marker::Send`.
    ///     std::thread::spawn(move || { println!("{:?}", val) });
    ///   }
    /// }
    ///
    /// #[cfg(not(feature = "sync"))]
    /// {
    ///   // If this code is compiled then `MaybeSend` dummy markerd implemented for all types.
    ///   maybe_sends(Rc::new(42));
    /// }
    /// ```
    pub trait MaybeSend {}

    /// All values are maybe sendable.
    impl<T> MaybeSend for T where T: ?Sized {}

    /// Marker trait that can represent nothing if feature "sync" is not enabled.
    /// Or be reexport of `std::marker::Sync` if "sync" feature is enabled.
    ///
    /// It is intended to be used as trait bound where `std::marker::Sync` bound
    /// is required only when application is compiled for multithreaded environment.\
    /// If "sync" feature is not enabled then this trait bound will *NOT* allow
    /// reference to the value to cross thread boundary or be used where sync
    /// value is expected.
    ///
    /// # Examples
    ///
    /// ```
    /// # use {maybe_sync::MaybeSync, std::{sync::Arc, fmt::Debug, cell::Cell}};
    ///
    /// fn maybe_shares<T: MaybeSync + Debug + 'static>(val: Arc<T>) {
    ///   #[cfg(feature = "sync")]
    ///   {
    ///     // If this code is compiled then `MaybeSync` is alias to `std::marker::Sync`.
    ///     std::thread::spawn(move || { println!("{:?}", val) });
    ///   }
    /// }
    ///
    /// #[cfg(not(feature = "sync"))]
    /// {
    ///   // If this code is compiled then `MaybeSync` dummy markerd implemented for all types.
    ///   maybe_shares(Arc::new(Cell::new(42)));
    /// }
    /// ```
    pub trait MaybeSync {}

    /// All values are maybe sync.
    impl<T> MaybeSync for T where T: ?Sized {}

    pub type BoxFuture<'a, T> = Pin<alloc::boxed::Box<dyn Future<Output = T> + 'a>>;

    pub type BoxStream<'a, T> = Pin<alloc::boxed::Box<dyn Stream<Item = T> + 'a>>;

    /// A pointer type which can be safely shared between threads
    /// when "sync" feature is enabled.\
    /// A pointer type which can be shared, but only within single thread
    /// where it was created when "sync" feature is not enabled.
    ///
    /// # Example
    ///
    /// ```
    /// # use {maybe_sync::{MaybeSend, Rc}, std::fmt::Debug};
    ///
    /// fn maybe_sends<T: MaybeSend + Debug + 'static>(val: T) {
    ///   #[cfg(feature = "sync")]
    ///   {
    ///     // If this code is compiled then `MaybeSend` is alias to `std::marker::Send`.
    ///     std::thread::spawn(move || { println!("{:?}", val) });
    ///   }
    /// }
    ///
    /// // Unlike `std::rc::Rc` this `maybe_sync::Rc<T>` would always
    /// // satisfy `MaybeSend` bound when `T: MaybeSend + MaybeSync`,
    /// // even if feature "sync" is enabeld.
    /// maybe_sends(Rc::new(42));
    /// ```
    pub type Rc<T> = alloc::rc::Rc<T>;

    /// Mutex implementation to use in conjunction with `MaybeSync` bound.
    ///
    /// A type alias to `parking_lot::Mutex` when "sync" feature is enabled.\
    /// A wrapper type around `std::cell::RefCell` when "sync" feature is not enabled.
    ///
    /// # Example
    ///
    /// ```
    /// # use {maybe_sync::{MaybeSend, Mutex}, std::{fmt::Debug, sync::Arc}};
    ///
    /// fn maybe_sends<T: MaybeSend + Debug + 'static>(val: Arc<Mutex<T>>) {
    ///   #[cfg(feature = "sync")]
    ///   {
    ///     // If this code is compiled then `MaybeSend` is alias to `std::marker::Send`,
    ///     // and `Mutex` is `parking_lot::Mutex`.
    ///     std::thread::spawn(move || { println!("{:?}", *val.lock()) });
    ///   }
    /// }
    ///
    /// // `maybe_sync::Mutex<T>` would always satisfy `MaybeSync` and `MaybeSend`
    /// // bounds when `T: MaybeSend`,
    /// // even if feature "sync" is enabeld.
    /// maybe_sends(Arc::new(Mutex::new(42)));
    /// ```
    #[repr(transparent)]
    #[derive(Debug, Default)]
    pub struct Mutex<T: ?Sized> {
        cell: RefCell<T>,
    }

    impl<T> Mutex<T> {
        /// Creates a new mutex in an unlocked state ready for use.
        pub fn new(value: T) -> Self {
            Mutex {
                cell: RefCell::new(value),
            }
        }
    }

    impl<T> Mutex<T>
    where
        T: ?Sized,
    {
        /// Acquires a mutex, blocking the current thread until it is able to do so.\
        /// This function will block the local thread until it is available to acquire the mutex.\
        /// Upon returning, the thread is the only thread with the mutex held.\
        /// An RAII guard is returned to allow scoped unlock of the lock.\
        /// When the guard goes out of scope, the mutex will be unlocked.\
        /// Attempts to lock a mutex in the thread which already holds the lock will result in a deadlock.
        pub fn lock(&self) -> RefMut<T> {
            self.cell.borrow_mut()
        }

        /// Attempts to acquire this lock.\
        /// If the lock could not be acquired at this time, then `None` is returned.\
        /// Otherwise, an RAII guard is returned.\
        /// The lock will be unlocked when the guard is dropped.\
        /// This function does not block.
        pub fn try_lock(&self) -> Option<RefMut<T>> {
            self.cell.try_borrow_mut().ok()
        }

        /// Returns a mutable reference to the underlying data.\
        /// Since this call borrows the `Mutex` mutably,\
        /// no actual locking needs to take place -
        /// the mutable borrow statically guarantees no locks exist.
        pub fn get_mut(&mut self) -> &mut T {
            self.cell.get_mut()
        }
    }

    /// A boolean type which can be safely shared between threads
    /// when "sync" feature is enabled.\
    /// A boolean type with non-threadsafe interior mutability
    /// when "sync" feature is not enabled.
    ///
    /// This type has the same in-memory representation as a bool.
    pub type AtomicBool = core::cell::Cell<bool>;

    /// A integer type which can be safely shared between threads
    /// when "sync" feature is enabled.\
    /// A integer type with non-threadsafe interior mutability
    /// when "sync" feature is not enabled.
    ///
    /// This type has the same in-memory representation as a i8.
    pub type AtomicI8 = core::cell::Cell<i8>;

    /// A integer type which can be safely shared between threads
    /// when "sync" feature is enabled.\
    /// A integer type with non-threadsafe interior mutability
    /// when "sync" feature is not enabled.
    ///
    /// This type has the same in-memory representation as a i16.
    pub type AtomicI16 = core::cell::Cell<i16>;

    /// A integer type which can be safely shared between threads
    /// when "sync" feature is enabled.\
    /// A integer type with non-threadsafe interior mutability
    /// when "sync" feature is not enabled.
    ///
    /// This type has the same in-memory representation as a i32.
    pub type AtomicI32 = core::cell::Cell<i32>;

    /// A integer type which can be safely shared between threads
    /// when "sync" feature is enabled.\
    /// A integer type with non-threadsafe interior mutability
    /// when "sync" feature is not enabled.
    ///
    /// This type has the same in-memory representation as a isize.
    pub type AtomicIsize = core::cell::Cell<isize>;

    /// A integer type which can be safely shared between threads
    /// when "sync" feature is enabled.\
    /// A integer type with non-threadsafe interior mutability
    /// when "sync" feature is not enabled.
    ///
    /// This type has the same in-memory representation as a i8.
    pub type AtomicU8 = core::cell::Cell<u8>;

    /// A integer type which can be safely shared between threads
    /// when "sync" feature is enabled.\
    /// A integer type with non-threadsafe interior mutability
    /// when "sync" feature is not enabled.
    ///
    /// This type has the same in-memory representation as a i16.
    pub type AtomicU16 = core::cell::Cell<u16>;

    /// A integer type which can be safely shared between threads
    /// when "sync" feature is enabled.\
    /// A integer type with non-threadsafe interior mutability
    /// when "sync" feature is not enabled.
    ///
    /// This type has the same in-memory representation as a i32.
    pub type AtomicU32 = core::cell::Cell<u32>;

    /// A integer type which can be safely shared between threads
    /// when "sync" feature is enabled.\
    /// A integer type with non-threadsafe interior mutability
    /// when "sync" feature is not enabled.
    ///
    /// This type has the same in-memory representation as a isize.
    pub type AtomicUsize = core::cell::Cell<usize>;

    /// A raw pointer type which can be safely shared between threads
    /// when "sync" feature is enabled.\
    /// A raw pointer type with non-threadsafe interior mutability
    /// when "sync" feature is not enabled.
    ///
    /// This type has the same in-memory representation as a isize.
    pub type AtomicPtr<T> = core::cell::Cell<*mut T>;
}

#[cfg(not(target_arch = "wasm32"))]
pub use sync::*;

#[cfg(target_arch = "wasm32")]
pub use unsync::*;

/// Expands to `dyn $traits` with `Send` marker trait
/// added when "sync" feature is enabled.
///
/// Expands to `dyn $traits` without `Send` marker trait
/// added "sync" feature is not enabled.
///
/// # Example
/// ```
/// # use maybe_sync::{MaybeSend, dyn_maybe_send};
/// fn foo<T: MaybeSend>(_: T) {}
/// // `x` will implement `MaybeSend` whether "sync" feature is enabled or not.
/// let x: Box<dyn_maybe_send!(std::future::Future<Output = u32>)> = Box::new(async move { 42 });
/// foo(x);
/// ```
#[cfg(not(target_arch = "wasm32"))]
#[macro_export]
macro_rules! dyn_maybe_send {
    ($($traits:tt)+) => {
        dyn $($traits)+ + Send
    };
}

/// Expands to `dyn $traits` with `Send` marker trait
/// added when "sync" feature is enabled.
///
/// Expands to `dyn $traits` without `Send` marker trait
/// added "sync" feature is not enabled.
///
/// # Example
/// ```
/// # use maybe_sync::{MaybeSend, dyn_maybe_send};
/// fn foo<T: MaybeSend>(_: T) {}
/// // `x` will implement `MaybeSend` whether "sync" feature is enabled or not.
/// let x: Box<dyn_maybe_send!(std::future::Future<Output = u32>)> = Box::new(async move { 42 });
/// foo(x);
/// ```
#[cfg(target_arch = "wasm32")]
#[macro_export]
macro_rules! dyn_maybe_send {
    ($($traits:tt)+) => {
        dyn $($traits)+
    };
}

/// Expands to `dyn $traits` with `Sync` marker trait
/// added when "sync" feature is enabled.
///
/// Expands to `dyn $traits` without `Sync` marker trait
/// added "sync" feature is not enabled.
///
/// # Example
/// ```
/// # use maybe_sync::{MaybeSync, dyn_maybe_sync};
/// fn foo<T: MaybeSync + ?Sized>(_: &T) {}
///
/// let x: &dyn_maybe_sync!(AsRef<str>) = &"qwerty";
/// // `x` will implement `MaybeSync` whether "sync" feature is enabled or not.
/// foo(x);
/// ```
#[cfg(not(target_arch = "wasm32"))]
#[macro_export]
macro_rules! dyn_maybe_sync {
    ($($traits:tt)+) => {
        dyn $($traits)+ + Sync
    };
}

/// Expands to `dyn $traits` with `Sync` marker trait
/// added when "sync" feature is enabled.
///
/// Expands to `dyn $traits` without `Sync` marker trait
/// added "sync" feature is not enabled.
///
/// # Example
/// ```
/// # use maybe_sync::{MaybeSync, dyn_maybe_sync};
/// fn foo<T: MaybeSync + ?Sized>(_: &T) {}
/// // `x` will implement `MaybeSync` whether "sync" feature is enabled or not.
/// let x: &dyn_maybe_sync!(AsRef<str>) = &"qwerty";
/// foo(x);
/// ```
#[cfg(target_arch = "wasm32")]
#[macro_export]
macro_rules! dyn_maybe_sync {
    ($($traits:tt)+) => {
        dyn $($traits)+
    };
}

/// Expands to `dyn $traits` with `Send` and `Sync` marker trait
/// added when "sync" feature is enabled.
///
/// Expands to `dyn $traits` without `Send` and `Sync` marker trait
/// added "sync" feature is not enabled.
///
/// # Example
/// ```
/// # use maybe_sync::{MaybeSend, MaybeSync, dyn_maybe_send_sync};
/// fn foo<T: MaybeSend + MaybeSync + ?Sized>(_: &T) {}
/// // `x` will implement `MaybeSend` and `MaybeSync` whether "sync" feature is enabled or not.
/// let x: &dyn_maybe_send_sync!(AsRef<str>) = &"qwerty";
/// foo(x);
/// ```
#[cfg(not(target_arch = "wasm32"))]
#[macro_export]
macro_rules! dyn_maybe_send_sync {
    ($($traits:tt)+) => {
        dyn $($traits)+ + Send + Sync
    };
}

/// Expands to `dyn $traits` with `Sync` marker trait
/// added when "sync" feature is enabled.
///
/// Expands to `dyn $traits` without `Sync` marker trait
/// added "sync" feature is not enabled.
///
/// # Example
/// ```
/// # use maybe_sync::{MaybeSend, MaybeSync, dyn_maybe_send_sync};
/// fn foo<T: MaybeSend + MaybeSync + ?Sized>(_: &T) {}
/// // `x` will implement `MaybeSend` and `MaybeSync` whether "sync" feature is enabled or not.
/// let x: &dyn_maybe_send_sync!(AsRef<str>) = &"qwerty";
/// foo(x);
/// ```
#[cfg(target_arch = "wasm32")]
#[macro_export]
macro_rules! dyn_maybe_send_sync {
    ($($traits:tt)+) => {
        dyn $($traits)+
    };
}
