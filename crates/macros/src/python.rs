use convert_case::{Case, Casing};
use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Fields, Meta, parse_macro_input};

pub fn py_string_enum(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let enum_name = &input.ident;

    // Extract rename_all rules from attributes
    let mut serde_rename_all = None;
    let mut strum_serialize_all = None;

    for attr in &input.attrs {
        if attr.path().is_ident("serde") {
            if let Meta::List(meta_list) = &attr.meta {
                let tokens_str = meta_list.tokens.to_string();
                let parts: Vec<&str> = tokens_str.split("=").collect();
                if parts.len() >= 2 {
                    let left = parts[0].trim();
                    if left == "rename_all" {
                        let right = parts[1].trim();
                        serde_rename_all = Some(right.trim_matches('"').to_string());
                    }
                }
            }
        } else if attr.path().is_ident("strum") {
            if let Meta::List(meta_list) = &attr.meta {
                let tokens_str = meta_list.tokens.to_string();
                let parts: Vec<&str> = tokens_str.split("=").collect();
                if parts.len() >= 2 {
                    let left = parts[0].trim();
                    if left == "serialize_all" {
                        let right = parts[1].trim();
                        strum_serialize_all = Some(right.trim_matches('"').to_string())
                    }
                }
            }
        }
    }

    // Prefer strum over serde since strum is used for Display/FromStr
    let rename_all = strum_serialize_all.or(serde_rename_all);

    let Data::Enum(data_enum) = &input.data else {
        panic!("PyStringEnum can only be derived for enums");
    };

    let mut literals = Vec::new();

    for variant in &data_enum.variants {
        if !matches!(variant.fields, Fields::Unit) {
            panic!("PyStringEnum only supports unit variants");
        }

        let variant_name = variant.ident.to_string();

        // Check for explicit rename attribute
        let mut explicit_rename = None;
        for attr in &variant.attrs {
            if attr.path().is_ident("serde") || attr.path().is_ident("strum") {
                if let Meta::List(meta_list) = &attr.meta {
                    let tokens_str = meta_list.tokens.to_string();
                    let parts: Vec<&str> = tokens_str.split("=").collect();
                    if parts.len() >= 2 {
                        let left = parts[0].trim();
                        if left == "rename" {
                            let right = parts[1].trim();
                            explicit_rename = Some(right.trim_matches('"').to_string());
                            break;
                        }
                    }
                }
            }
        }

        let literal = if let Some(renamed) = explicit_rename {
            renamed
        } else if let Some(rule) = &rename_all {
            apply_rename_rule(&variant_name, rule)
        } else {
            variant_name
        };

        literals.push(literal);
    }

    let literal_strings: Vec<_> = literals.iter().map(|lit| quote! { #lit }).collect();

    let error_message = literals.join(", ");

    let expanded = quote! {
        #[cfg(feature = "python")]
        const _: () = {
            use pyo3::{
                Bound, Borrowed, FromPyObject, IntoPyObject, PyAny, PyErr, PyResult, Python,
                exceptions::PyValueError,
                types::{PyAnyMethods, PyString},
            };
            use pyo3_stub_gen::{PyStubType, TypeInfo};

            impl<'py> IntoPyObject<'py> for #enum_name {
                type Target = PyString;
                type Output = Bound<'py, PyString>;
                type Error = PyErr;

                fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
                    Ok(PyString::new(py, &self.to_string()))
                }
            }

            impl<'a, 'py> FromPyObject<'a, 'py> for #enum_name {
                type Error = PyErr;

                fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
                    let s: &str = ob.extract()?;
                    s.parse::<#enum_name>()
                        .map_err(|_| PyValueError::new_err(format!(
                            "Invalid {}: {}. Expected one of: {}",
                            stringify!(#enum_name),
                            s,
                            #error_message
                        )))
                }
            }

            impl PyStubType for #enum_name {
                fn type_output() -> TypeInfo {
                    let mut import = std::collections::HashSet::new();
                    import.insert("typing".into());

                    // Build the literal string dynamically with the renamed values
                    let literals = vec![#(#literal_strings),*];
                    let literal_str = literals
                        .iter()
                        .map(|s| format!(r#""{}""#, s))
                        .collect::<Vec<_>>()
                        .join(", ");

                    TypeInfo {
                        name: format!(r#"typing.Literal[{}]"#, literal_str),
                        import,
                    }
                }
            }
        };
    };

    TokenStream::from(expanded)
}

fn apply_rename_rule(name: &str, rule: &str) -> String {
    match rule {
        "lowercase" => name.to_case(Case::Lower),
        "UPPERCASE" => name.to_case(Case::Upper),
        "PascalCase" => name.to_case(Case::Pascal),
        "camelCase" => name.to_case(Case::Camel),
        "snake_case" => name.to_case(Case::Snake),
        "SCREAMING_SNAKE_CASE" => name.to_case(Case::UpperSnake),
        "kebab-case" => name.to_case(Case::Kebab),
        "SCREAMING-KEBAB-CASE" => name.to_case(Case::UpperKebab),
        _ => name.to_string(),
    }
}
