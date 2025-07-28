use std::ffi::{CStr, CString, c_char};

/// Converts a C-style `const char*` to a Rust `&str`.
///
/// # Arguments
/// * `v` - A pointer to a null-terminated C string (`const char*`).
///
/// # Returns
/// * `Ok(&str)` if the pointer is valid and UTF-8 encoded.
/// * `Err(())` if the pointer is null or the string is not valid UTF-8.
///
/// # Safety
/// This function assumes the pointer is valid and points to a null-terminated string.
/// It performs an unsafe dereference internally.
///
/// # Example (C-style)
/// ```c
/// const char* msg = "hello";
/// rust_fn(msg);
/// ```
pub fn from_const_char<'a>(v: *const c_char) -> Result<&'a str, ()> {
    if v.is_null() {
        Err(())
    } else {
        match unsafe { CStr::from_ptr(v) }.to_str() {
            Ok(s) => Ok(s),
            Err(_) => Err(()),
        }
    }
}

/// Converts a Rust `&str` to a heap-allocated C-style `*mut char`.
///
/// # Arguments
/// * `v` - A Rust string slice (`&str`) to convert.
///
/// # Returns
/// * A raw pointer (`*mut c_char`) to a null-terminated C string on the heap.
/// * If the input contains interior null bytes (`\0`), returns a null pointer (`std::ptr::null_mut()`).
///
/// # Safety
/// The returned pointer must be freed by the caller using `CString::from_raw` to avoid memory leaks.
/// This is FFI-safe and suitable for interop with C or C++ code.
///
/// # Example (C++ side)
/// ```cpp
/// char* s = to_char("hello");
/// ... use s ...
/// free_from_rust(s); // You must provide a corresponding free function in Rust
/// ```
pub fn to_char(v: &str) -> *mut c_char {
    match CString::new(v) {
        Ok(cstring) => cstring.into_raw(),
        Err(_) => std::ptr::null_mut(), // If input has interior nulls
    }
}
