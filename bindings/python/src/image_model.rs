//! `ImageModel`, built from a desc and run on a prompt.

use std::sync::Arc;

use ailoy::model::{ImageModelOptions, InferImageModel, create_image_model};
use pyo3::prelude::*;
use pyo3_async_runtimes::tokio::future_into_py;

use crate::{
    agent::{OnRuntime, item},
    convert::from_py,
    error,
};

/// An image model built from its desc, run on a prompt.
#[pyclass(name = "ImageModel", module = "ailoy", frozen)]
pub struct PyImageModel(OnRuntime<Arc<dyn InferImageModel>>);

impl PyImageModel {
    fn model(&self) -> Arc<dyn InferImageModel> {
        self.0
            .0
            .clone()
            .expect("a model is only let go of when dropped")
    }
}

#[pymethods]
impl PyImageModel {
    /// The image model `desc` names, `<org>/<name>`. Awaitable.
    #[staticmethod]
    fn create(py: Python<'_>, desc: String) -> PyResult<Bound<'_, PyAny>> {
        future_into_py(py, async move {
            let model = create_image_model(&desc).await.map_err(error::anyhow)?;
            Ok(PyImageModel(OnRuntime(Some(model))))
        })
    }

    /// Generate images from `prompt`. Awaitable; answers with at least one image, and raises
    /// for a call that ends with none.
    #[pyo3(signature = (prompt, *, options = None))]
    fn generate<'py>(
        &self,
        py: Python<'py>,
        prompt: String,
        options: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let options = match options {
            Some(options) => from_py(&options, "image model options")?,
            None => ImageModelOptions::default(),
        };
        let model = self.model();
        future_into_py(
            py,
            async move { item(model.generate(&prompt, &options).await) },
        )
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyImageModel>()?;
    Ok(())
}
