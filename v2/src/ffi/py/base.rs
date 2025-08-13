use pyo3::{PyClass, prelude::*};

pub trait PyWrapper: PyClass<BaseType = PyAny> + Send + 'static {
    type Inner: Send;

    fn from_inner(inner: Self::Inner) -> Self;
}
