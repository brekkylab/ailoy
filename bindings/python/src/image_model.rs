//! `ImageModel`, built from a desc, or around an image model of Python's own, and run on a
//! prompt.
//!
//! A model of Python's own is any object with `generate(prompt, options)`, returning an
//! `ImageOutput` dict or an awaitable of one. `options` is the dict `ImageModel.generate` was
//! given, or `{}` without any; its keys are the model's to read. `generate` runs off the event
//! loop when synchronous and on the caller's when async.

use std::{collections::BTreeMap, sync::Arc};

use ailoy::{
    datatype::Value,
    model::{ImageOptions, ImageOutput, InferImageModel, create_image_model},
};
use futures::future::BoxFuture;
use pyo3::{exceptions::PyValueError, prelude::*};
use pyo3_async_runtimes::tokio::future_into_py;

use crate::{
    agent::{OnRuntime, item},
    convert::{from_py, to_py},
    error,
    tool::call_py,
};

/// An image model built from its desc, or around one of Python's own, run on a prompt.
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

    /// An `ImageModel` that runs `model`, an object of Python's own with `generate`, as the
    /// module says.
    #[staticmethod]
    fn custom(model: Bound<'_, PyAny>) -> PyResult<PyImageModel> {
        if !model.hasattr("generate")? {
            return Err(PyValueError::new_err(
                "an image model has a `generate` method",
            ));
        }
        let model: Arc<dyn InferImageModel> = Arc::new(PyModel(Arc::new(model.unbind())));
        Ok(PyImageModel(OnRuntime(Some(model))))
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
        // Any keys: which a model takes is the model's to say.
        let options: Option<BTreeMap<String, Value>> = options
            .map(|options| from_py(&options, "image model options"))
            .transpose()?;
        let model = self.model();
        future_into_py(py, async move {
            let options = options.as_ref().map(|o| o as &dyn ImageOptions);
            item(model.generate(&prompt, options).await)
        })
    }
}

/// An image model of Python's own.
struct PyModel(Arc<Py<PyAny>>);

impl InferImageModel for PyModel {
    fn generate(
        &self,
        prompt: &str,
        options: Option<&dyn ImageOptions>,
    ) -> BoxFuture<'static, anyhow::Result<ImageOutput>> {
        let model = self.0.clone();
        let prompt = prompt.to_owned();
        // No options reach the model as an empty object, so it need not tell the two apart.
        let options = options
            .map(|o| o.read::<Value>())
            .transpose()
            .map(|o| o.unwrap_or_else(Value::object_empty));
        Box::pin(async move {
            let options = options?;
            let returned = call_py(move |py| {
                let options = to_py(py, &options)?;
                model.call_method1(py, "generate", (prompt, options))
            })
            .await?;
            Ok(Python::attach(|py| {
                from_py::<ImageOutput>(returned.bind(py), "an image output")
            })?)
        })
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyImageModel>()?;
    Ok(())
}
