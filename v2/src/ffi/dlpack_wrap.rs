use crate::ffi::util::*;
use crate::ffi::{DLManagedTensorVersioned, DLPackTensor, ManagedTensor};

unsafe impl Send for ffi::DLDevice {}

unsafe impl Send for ffi::DLPackTensor {}

impl DLPackTensor {
    // /// from raw DLManagedTensorVersioned pointer
    // pub unsafe fn from_raw(ptr: *mut DLManagedTensorVersioned) -> Result<Self> {
    //     let managed = unsafe { ffi::create_managed_tensor(ptr) }?;
    //     Ok(Self { inner: managed })
    // }

    /// 1-dimensional float32 Tensor to Vec<f32>
    pub fn to_vec_f32(&self) -> Result<Vec<f32>> {
        let dimension = self.inner.get_dimension();
        if dimension == -1 {
            bail!("Tensor is not 1D.");
        }
        if !self.inner.is_cpu_tensor() {
            bail!("GPU tensors not yet supported. CPU tensors only");
        }

        if self.inner.has_float_dtype(32) {
            let data_ptr = self.inner.get_data_ptr_f32();
            if data_ptr.is_null() {
                bail!("Tensor data pointer is null");
            }
            let vec = unsafe { std::slice::from_raw_parts(data_ptr, dimension as usize).to_vec() };
            Ok(vec)
        } else if self.inner.has_float_dtype(16) {
            let data_ptr = self.inner.get_data_ptr_u16();
            if data_ptr.is_null() {
                bail!("Tensor data pointer is null");
            }
            let vec = unsafe { std::slice::from_raw_parts(data_ptr, dimension as usize).to_vec() }
                .into_iter()
                .map(|val| util::f16_to_f32(val))
                .collect();
            Ok(vec)
        } else {
            bail!("Tensor has unsupported dtype.");
        }
    }

    // /// Rust type to C++ UniquePtr
    // fn into_inner(self) -> cxx::UniquePtr<ffi::ManagedTensor> {
    //     self.inner
    // }
}
