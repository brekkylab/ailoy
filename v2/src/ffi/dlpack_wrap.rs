use anyhow::bail;

use crate::{
    ffi::cxx_bridge::{DLDevice, DLPackTensor},
    utils::float16,
};

unsafe impl Send for DLDevice {}

unsafe impl Send for DLPackTensor {}

impl DLPackTensor {
    /// 1-dimensional float32 Tensor to Vec<f32>
    pub fn to_vec_f32(&self) -> anyhow::Result<Vec<f32>> {
        let managed_tensor = self.inner.as_ref().unwrap();
        let dimension = managed_tensor.get_dimension();
        if dimension == -1 {
            bail!("Tensor is not 1D.");
        }
        if !managed_tensor.is_cpu_tensor() {
            bail!("GPU tensors not yet supported. CPU tensors only");
        }

        if managed_tensor.has_float_dtype(32) {
            let data_ptr = managed_tensor.get_data_ptr_f32();
            if data_ptr.is_null() {
                bail!("Tensor data pointer is null");
            }
            let vec = unsafe { std::slice::from_raw_parts(data_ptr, dimension as usize).to_vec() };
            Ok(vec)
        } else if managed_tensor.has_float_dtype(16) {
            let data_ptr = managed_tensor.get_data_ptr_u16();
            if data_ptr.is_null() {
                bail!("Tensor data pointer is null");
            }
            let vec = unsafe { std::slice::from_raw_parts(data_ptr, dimension as usize).to_vec() }
                .into_iter()
                .map(|val| float16::f16_to_f32(val))
                .collect();
            Ok(vec)
        } else {
            bail!("Tensor has unsupported dtype.");
        }
    }
}
