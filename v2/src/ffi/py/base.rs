use pyo3::{
    IntoPyObjectExt,
    exceptions::PyRuntimeError,
    prelude::*,
    types::{PyDict, PyList},
};
use serde_json::{Map, Value};

#[allow(unused)]
pub fn json_value_to_py_object(py: Python, value: &Value) -> PyResult<Py<PyAny>> {
    match value {
        Value::Null => Ok(py.None()),
        Value::Bool(b) => Ok(b.into_py_any(py).unwrap()),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.into_py_any(py).unwrap())
            } else if let Some(u) = n.as_u64() {
                Ok(u.into_py_any(py).unwrap())
            } else if let Some(f) = n.as_f64() {
                Ok(f.into_py_any(py).unwrap())
            } else {
                // Fallback: convert to string if number format is unexpected
                Ok(n.to_string().into_py_any(py).unwrap())
            }
        }
        Value::String(s) => Ok(s.into_py_any(py).unwrap()),
        Value::Array(arr) => {
            let py_list = PyList::empty(py);
            for item in arr {
                let py_item = json_value_to_py_object(py, item)?;
                py_list.append(py_item)?;
            }
            Ok(py_list.into_py_any(py).unwrap())
        }
        Value::Object(obj) => {
            let py_dict = PyDict::new(py);
            for (key, val) in obj {
                let py_val = json_value_to_py_object(py, val)?;
                py_dict.set_item(key, py_val)?;
            }
            Ok(py_dict.into_py_any(py).unwrap())
        }
    }
}

#[allow(unused)]
pub fn json_to_pydict<'py>(py: Python<'py>, value: &Value) -> PyResult<Option<Bound<'py, PyDict>>> {
    match value {
        Value::Object(obj) => {
            let py_dict = PyDict::new(py);
            for (key, val) in obj {
                let py_val = json_value_to_py_object(py, val)?;
                py_dict.set_item(key, py_val)?;
            }
            Ok(Some(py_dict))
        }
        Value::Null => Ok(None),
        _ => Err(pyo3::exceptions::PyTypeError::new_err(
            "JSON value is not an object, cannot convert to PyDict",
        )),
    }
}

// Conversion from PyDict back to serde_json::Value
#[allow(unused)]
fn py_object_to_json_value(py: Python, obj: &Py<PyAny>) -> PyResult<Value> {
    if obj.is_none(py) {
        Ok(Value::Null)
    } else if let Ok(b) = obj.extract::<bool>(py) {
        Ok(Value::Bool(b))
    } else if let Ok(i) = obj.extract::<i64>(py) {
        Ok(Value::Number(serde_json::Number::from(i)))
    } else if let Ok(f) = obj.extract::<f64>(py) {
        if let Some(num) = serde_json::Number::from_f64(f) {
            Ok(Value::Number(num))
        } else {
            Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Invalid float value: {}",
                f
            )))
        }
    } else if let Ok(s) = obj.extract::<String>(py) {
        Ok(Value::String(s))
    } else if let Ok(py_list) = obj.downcast_bound::<PyList>(py) {
        let mut arr = Vec::new();
        for item in py_list.iter() {
            let json_val = py_object_to_json_value(py, &item.into())?;
            arr.push(json_val);
        }
        Ok(Value::Array(arr))
    } else if let Ok(py_dict) = obj.downcast_bound::<PyDict>(py) {
        let mut map = serde_json::Map::new();
        for (key, value) in py_dict.iter() {
            let key_str = key.extract::<String>()?;
            let json_val = py_object_to_json_value(py, &value.into())?;
            map.insert(key_str, json_val);
        }
        Ok(Value::Object(map))
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "Unsupported Python type: {}",
            obj.bind(py).get_type().name()?
        )))
    }
}

#[allow(unused)]
// Main conversion function from PyDict to serde_json::Map
pub fn pydict_to_json(py: Python, py_dict: &Bound<PyDict>) -> PyResult<Map<String, Value>> {
    let mut map = serde_json::Map::new();
    for (key, value) in py_dict.iter() {
        let key_str = key.extract::<String>()?;
        let json_val = py_object_to_json_value(py, &value.into())?;
        map.insert(key_str, json_val);
    }
    Ok(map)
}

pub fn await_future<T, E: ToString + std::any::Any>(
    fut: impl Future<Output = Result<T, E>>,
) -> PyResult<T> {
    let rt = pyo3_async_runtimes::tokio::get_runtime();
    let result = rt.block_on(fut).map_err(|e| {
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
