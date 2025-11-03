use std::fmt;

use serde::{Deserialize, Serialize};

use crate::value::Value;

/// Describes a **tool** (or function) that a language model can invoke.
///
/// `ToolDesc` defines the schema, behavior, and input/output specification of a callable
/// external function, allowing an LLM to understand how to use it.
///
/// The primary role of this struct is to describe to the LLM what a *tool* does,
/// how it can be invoked, and what input (`parameters`) and output (`returns`) schemas it expects.
///
/// The format follows the same **schema conventions** used by Hugging Face’s
/// `transformers` library, as well as APIs such as *OpenAI* and *Anthropic*.
/// The `parameters` and `returns` fields are typically defined using **JSON Schema**.
///
/// We provide a builder [`ToolDescBuilder`] helper for convenient and fluent construction.
/// Please refer to [`ToolDescBuilder`].
///
/// # Example
/// ```rust
/// use crate::value::{ToolDescBuilder, to_value};
///
/// let desc = ToolDescBuilder::new("temperature")
///     .description("Get the current temperature for a given city")
///     .parameters(to_value!({
///         "type": "object",
///         "properties": {
///             "location": {
///                 "type": "string",
///                 "description": "The city name"
///             },
///             "unit": {
///                 "type": "string",
///                 "description": "Temperature unit (default: Celsius)",
///                 "enum": ["Celsius", "Fahrenheit"]
///             }
///         },
///         "required": ["location"]
///     }))
///     .returns(to_value!({
///         "type": "number"
///     }))
///     .build();
///
/// assert_eq!(desc.name, "temperature");
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "python", pyo3_stub_gen_derive::gen_stub_pyclass)]
#[cfg_attr(feature = "python", pyo3::pyclass(module = "ailoy._core"))]
#[cfg_attr(feature = "nodejs", napi_derive::napi(object))]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(feature = "wasm", tsify(into_wasm_abi, from_wasm_abi))]
pub struct ToolDesc {
    /// The unique name of the tool or function.
    pub name: String,

    /// A natural-language description of what the tool does.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// A [`Value`] describing the JSON Schema of the expected parameters.
    /// Typically an object schema such as `{ "type": "object", "properties": ... }`.
    pub parameters: Value,

    /// An optional [`Value`] that defines the return value schema.  
    /// If omitted, the tool is assumed to return free-form text or JSON.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub returns: Option<Value>,
}

impl ToolDesc {
    pub fn new(
        name: String,
        description: Option<String>,
        parameters: Value,
        returns: Option<Value>,
    ) -> Self {
        ToolDesc {
            name,
            description,
            parameters,
            returns,
        }
    }
}

impl fmt::Display for ToolDesc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = serde_json::to_string(self).map_err(|_| fmt::Error)?;
        write!(f, "ToolDesc {}", s)
    }
}

/// A builder for constructing [`ToolDesc`] objects.
///
/// Provides a fluent, chainable API for creating a tool description safely and clearly.
/// If no `parameters` are provided, it defaults to [`Value::Null`].
///
/// # Example
/// ```rust
/// use crate::value::{ToolDescBuilder, Value};
///
/// let tool = ToolDescBuilder::new("weather")
///     .description("Fetch current weather information")
///     .parameters(Value::Null)
///     .returns(Value::Null)
///     .build();
///
/// assert_eq!(tool.name, "weather");
/// ```
#[derive(Clone, Debug)]
pub struct ToolDescBuilder {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Option<Value>,
    pub returns: Option<Value>,
}

impl ToolDescBuilder {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: None,
            parameters: None,
            returns: None,
        }
    }

    pub fn description(self, desc: impl Into<String>) -> Self {
        Self {
            name: self.name,
            description: Some(desc.into()),
            parameters: self.parameters,
            returns: self.returns,
        }
    }

    pub fn parameters(self, param: impl Into<Value>) -> Self {
        Self {
            name: self.name,
            description: self.description,
            parameters: Some(param.into()),
            returns: self.returns,
        }
    }

    pub fn returns(self, ret: impl Into<Value>) -> Self {
        Self {
            name: self.name,
            description: self.description,
            parameters: self.parameters,
            returns: Some(ret.into()),
        }
    }

    pub fn build(self) -> ToolDesc {
        ToolDesc {
            name: self.name,
            description: self.description,
            parameters: match self.parameters {
                Some(p) => p,
                None => Value::Null,
            },
            returns: self.returns,
        }
    }
}

#[cfg(feature = "python")]
pub(crate) mod py {
    use pyo3::{Py, PyResult, Python, prelude::*, pymethods, types::PyDict};
    use pyo3_stub_gen::derive::*;

    use super::*;
    use crate::ffi::py::base::{python_to_value, value_to_python};

    #[gen_stub_pymethods]
    #[pymethods]
    impl ToolDesc {
        #[new]
        #[pyo3(signature = (name, description, parameters, *, returns=None))]
        fn __new__(
            py: Python<'_>,
            name: String,
            description: Option<String>,
            parameters: Py<PyDict>,
            returns: Option<Py<PyDict>>,
        ) -> PyResult<Self> {
            let parameters = python_to_value(parameters.bind(py))?;
            let returns = if let Some(returns) = returns {
                Some(python_to_value(returns.bind(py))?)
            } else {
                None
            };
            anyhow::Ok(Self::new(
                name,
                description,
                parameters.into(),
                returns.map(|returns| returns.into()),
            ))
            .map_err(Into::into)
        }

        pub fn __repr__(&self) -> String {
            let returns_display = self
                .returns
                .clone()
                .and_then(|v| serde_json::to_string(&v).ok());

            if let Some(returns_display) = returns_display {
                format!(
                    "ToolDesc(name=\"{}\", description={}, parameters={}, returns={})",
                    self.name,
                    self.description.clone().unwrap_or_default(),
                    serde_json::to_string(&self.parameters).expect("Invalid parameters"),
                    returns_display,
                )
            } else {
                format!(
                    "ToolDesc(name=\"{}\", description={}, parameters={})",
                    self.name,
                    self.description.clone().unwrap_or_default(),
                    serde_json::to_string(&self.parameters).expect("Invalid parameters"),
                )
            }
        }

        #[getter]
        fn name(&self) -> String {
            self.name.clone()
        }

        #[getter]
        fn description(&self) -> Option<String> {
            self.description.clone()
        }

        #[getter]
        fn parameters<'py>(&self, py: Python<'py>) -> Bound<'py, PyDict> {
            value_to_python(py, &self.parameters)
                .unwrap()
                .cast_into()
                .unwrap()
        }

        #[getter]
        fn returns<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyDict>> {
            self.returns
                .as_ref()
                .and_then(|returns| value_to_python(py, returns).ok()?.cast_into().ok())
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::to_value;

    #[test]
    fn simple_tool_description_serde() {
        let desc = ToolDescBuilder::new("temperature")
            .description("Get current temperature")
            .parameters(to_value!({
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city name"
                    },
                    "unit": {
                        "type": "string",
                        "description": "Default: Celsius",
                        "enum": ["Celsius", "Fahrenheit"]
                    }
                },
                "required": ["location"]
            }))
            .build();

        let serialized = {
            let expected = r#"{"name":"temperature","description":"Get current temperature","parameters":{"type":"object","properties":{"location":{"type":"string","description":"The city name"},"unit":{"type":"string","description":"Default: Celsius","enum":["Celsius","Fahrenheit"]}},"required":["location"]},"returns":{"type":"number"}}"#;
            let actual = serde_json::to_string(&desc).unwrap();
            assert_eq!(expected, actual);
            actual
        };
        let recovered: ToolDesc = serde_json::from_str(&serialized).unwrap();
        assert_eq!(desc, recovered);
    }
}
