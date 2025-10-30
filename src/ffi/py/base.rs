use indexmap::IndexMap;
use ordered_float::OrderedFloat;
use pyo3::{
    IntoPyObjectExt,
    exceptions::{PyRuntimeError, PyTypeError},
    prelude::*,
    types::{PyAny, PyBool, PyDict, PyFloat, PyList, PySequence, PyString, PyTuple},
};

use crate::value::Value;

pub fn value_to_python<'py>(py: Python<'py>, value: &Value) -> PyResult<Bound<'py, PyAny>> {
    match value {
        Value::Null => py.None().into_bound_py_any(py),
        Value::Bool(b) => PyBool::new(py, *b).into_bound_py_any(py),
        Value::Unsigned(u) => u.into_bound_py_any(py),
        Value::Integer(i) => i.into_bound_py_any(py),
        Value::Float(f) => PyFloat::new(py, f.0).into_bound_py_any(py),
        Value::String(s) => PyString::new(py, s).into_bound_py_any(py),
        Value::Array(arr) => {
            let py_list = PyList::empty(py);
            for item in arr {
                py_list.append(value_to_python(py, item)?)?;
            }
            py_list.into_bound_py_any(py)
        }
        Value::Object(map) => {
            let py_dict = PyDict::new(py);
            for (key, val) in map {
                py_dict.set_item(key, value_to_python(py, val)?)?;
            }
            py_dict.into_bound_py_any(py)
        }
    }
}

pub fn python_to_value(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    if obj.is_none() {
        Ok(Value::Null)
    } else if let Ok(b) = obj.extract::<bool>() {
        Ok(Value::Bool(b))
    } else if let Ok(int_val) = obj.extract::<i128>() {
        let ret = if let Ok(i) = i64::try_from(int_val) {
            Value::Integer(i)
        } else if int_val >= 0
            && let Ok(u) = u64::try_from(int_val)
        {
            Value::Unsigned(u)
        } else {
            return Err(PyTypeError::new_err("int out of supported range (i64/u64)"));
        };
        Ok(ret)
    } else if let Ok(f) = obj.extract::<f64>() {
        Ok(Value::Float(OrderedFloat(f)))
    } else if let Ok(s) = obj.extract::<String>() {
        Ok(Value::String(s))
    } else if let Ok(list) = obj.cast::<PyList>() {
        let mut arr = Vec::with_capacity(list.len() as usize);
        for item in list.iter() {
            arr.push(python_to_value(&item)?);
        }
        Ok(Value::Array(arr))
    } else if let Ok(tup) = obj.cast::<PyTuple>() {
        let mut arr = Vec::with_capacity(tup.len() as usize);
        for item in tup.iter() {
            arr.push(python_to_value(&item)?);
        }
        Ok(Value::Array(arr))
    } else if let Ok(seq) = obj.cast::<PySequence>() {
        let mut arr = Vec::with_capacity(seq.len()? as usize);
        for item in seq.try_iter()? {
            arr.push(python_to_value(&item?)?);
        }
        Ok(Value::Array(arr))
    } else if let Ok(dict) = obj.cast::<PyDict>() {
        let mut map = IndexMap::new();
        for (key, val) in dict.iter() {
            let key_str = key.extract::<String>()?;
            map.insert(key_str, python_to_value(&val)?);
        }
        Ok(Value::Object(map))
    } else {
        Ok(Value::String(obj.str()?.to_string()))
    }
}

pub fn await_future<T: Send, E: ToString + std::any::Any + Send>(
    py: Python<'_>,
    fut: impl Future<Output = Result<T, E>> + Send,
) -> PyResult<T> {
    let rt = pyo3_async_runtimes::tokio::get_runtime();
    // Release the GIL while waiting
    let result = py.detach(|| rt.block_on(fut)).map_err(|e| {
        if std::any::TypeId::of::<E>() == std::any::TypeId::of::<PyErr>() {
            // PyErr is returned as-is
            let any_err = Box::new(e) as Box<dyn std::any::Any>;
            let py_err = *any_err.downcast::<PyErr>().unwrap();
            py_err
        } else {
            // Other errors are converted to PyRuntimeError
            PyRuntimeError::new_err(e.to_string())
        }
    });
    result
}

pub trait PyRepr<T>
where
    T: std::fmt::Display,
{
    fn __repr__(&self) -> String;
}

impl<T> PyRepr<T> for T
where
    T: std::fmt::Display,
{
    fn __repr__(&self) -> String {
        format!(r#""{}""#, self)
    }
}

impl<T> PyRepr<T> for Option<T>
where
    T: std::fmt::Display,
{
    fn __repr__(&self) -> String {
        match self {
            Some(val) => format!(r#""{}""#, val),
            None => "None".into(),
        }
    }
}
