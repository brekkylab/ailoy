use serde::{Deserialize, Serialize};

use crate::value::Value;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "python", pyo3_stub_gen_derive::gen_stub_pyclass)]
#[cfg_attr(feature = "python", pyo3::pyclass)]
#[cfg_attr(feature = "nodejs", napi_derive::napi(object))]
#[cfg_attr(
    feature = "wasm",
    wasm_bindgen::prelude::wasm_bindgen(getter_with_clone)
)]
pub struct ToolDesc {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Value,
    pub returns: Option<Value>,
}

impl ToolDesc {}

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
