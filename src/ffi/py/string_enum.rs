#[macro_export]
macro_rules! impl_py_string_enum {
    ($enum_name:ident, [$($variant:ident => $literal:literal),+ $(,)?]) => {
        impl<'py> IntoPyObject<'py> for $enum_name {
            type Target = PyString;
            type Output = Bound<'py, PyString>;
            type Error = PyErr;

            fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
                Ok(PyString::new(py, &self.to_string()))
            }
        }

        impl<'py> FromPyObject<'py> for $enum_name {
            fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
                let s: &str = ob.extract()?;
                s.parse::<$enum_name>()
                    .map_err(|_| PyValueError::new_err(format!(
                        "Invalid {}: {}. Expected one of: {}",
                        stringify!($enum_name),
                        s,
                        vec![$($literal),+].join(", ")
                    )))
            }
        }

        impl PyStubType for $enum_name {
            fn type_output() -> TypeInfo {
                let mut import = std::collections::HashSet::new();
                import.insert("typing".into());

                TypeInfo {
                    name: format!(
                        r#"typing.Literal[{}]"#,
                        vec![$($literal),+]
                            .iter()
                            .map(|s| format!(r#""{}""#, s))
                            .collect::<Vec<_>>()
                            .join(", ")
                    ),
                    import,
                }
            }
        }
    };
}
