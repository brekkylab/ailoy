//! `ImageModel`, and the per-org providers image models are built through, in a table apart
//! from the language models'.
//!
//! An org's provider is a backend (`"api"`, the org's own API, is the one image models run
//! through) or a factory of Python's own, as [`ImageModelProvider::custom`] is in Rust: called
//! with the org and the name after it (`None` for a desc naming only the org), it returns a
//! model. That is an `ImageModel`, used as is, or any object with `generate(prompt, options)`,
//! returning an `ImageOutput` dict or an awaitable of one, with the options in their dict form.
//!
//! The factory is called synchronously, whenever `ImageModel.create` builds a model of the org;
//! `generate` off the event loop when synchronous and on the caller's when async, as tools are.

use std::sync::Arc;

use ailoy::model::{
    Backend, ImageModelOptions, ImageModelProvider, ImageOutput, InferImageModel,
    create_image_model, get_image_model_provider as get_image_model_provider_rs,
    remove_image_model_provider as remove_image_model_provider_rs,
    set_image_model_provider as set_image_model_provider_rs,
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
    /// The image model `desc` names, `<org>/<name>`, built through the provider set for
    /// `<org>`. Awaitable.
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

/// An image model of Python's own, as a factory returned it.
struct PyModel(Arc<Py<PyAny>>);

impl InferImageModel for PyModel {
    fn generate(
        &self,
        prompt: &str,
        options: &ImageModelOptions,
    ) -> BoxFuture<'static, anyhow::Result<ImageOutput>> {
        let model = self.0.clone();
        let (prompt, options) = (prompt.to_owned(), options.clone());
        Box::pin(async move {
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

/// The image models of `org` from what `factory` returns for it.
fn custom(factory: Py<PyAny>) -> ImageModelProvider {
    ImageModelProvider::custom(move |org, name| {
        Python::attach(|py| -> PyResult<Arc<dyn InferImageModel>> {
            let model = factory.call1(py, (org, name))?;
            if let Ok(model) = model.bind(py).cast::<PyImageModel>() {
                return Ok(model.get().model());
            }
            Ok(Arc::new(PyModel(Arc::new(model))))
        })
        .map_err(Into::into)
    })
}

/// Build the image models of `org` through `provider`: a backend's name, or a factory, as the
/// module says. Replaces the provider the org had.
#[pyfunction]
fn set_image_model_provider(org: String, provider: Bound<'_, PyAny>) -> PyResult<()> {
    let provider = if provider.is_instance_of::<pyo3::types::PyString>() {
        ImageModelProvider::from(from_py::<Backend>(&provider, "a backend")?)
    } else if provider.is_callable() {
        custom(provider.unbind())
    } else {
        return Err(PyValueError::new_err(
            "an image model provider is a backend's name or a factory of models",
        ));
    };
    set_image_model_provider_rs(org, provider);
    Ok(())
}

/// How the image model `desc` names (`"<org>/<name>"`, or `"<org>"`) is built: a backend's
/// name, `"custom"` for a factory, or `None` when its org has no provider.
#[pyfunction]
fn get_image_model_provider(desc: String) -> Option<String> {
    get_image_model_provider_rs(&desc).map(|provider| match provider {
        ImageModelProvider::Backend(backend) => backend.to_string(),
        ImageModelProvider::Custom(_) => "custom".to_string(),
    })
}

/// Stop building the image models of `org`. Answers whether it had a provider.
#[pyfunction]
fn remove_image_model_provider(org: String) -> bool {
    remove_image_model_provider_rs(&org).is_some()
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyImageModel>()?;
    m.add_function(wrap_pyfunction!(set_image_model_provider, m)?)?;
    m.add_function(wrap_pyfunction!(get_image_model_provider, m)?)?;
    m.add_function(wrap_pyfunction!(remove_image_model_provider, m)?)?;
    Ok(())
}
