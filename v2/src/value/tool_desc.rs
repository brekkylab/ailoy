use serde::{Deserialize, Serialize};

use crate::value::Value;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "python", pyo3_stub_gen_derive::gen_stub_pyclass)]
#[cfg_attr(feature = "python", pyo3::pyclass)]
pub struct ToolDesc {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

impl ToolDesc {
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: impl Into<Value>,
    ) -> Result<Self, String> {
        let parameters = parameters.into();
        let parameters_json: serde_json::Value = parameters.clone().into();
        jsonschema::validator_for(&parameters_json)
            .map_err(|e| format!("parameters is not a valid jsonschema: {}", e.to_string()))?;
        Ok(Self {
            name: name.into(),
            description: description.into(),
            parameters,
        })
    }
}

#[cfg(test)]
mod test {
    use serde_json::json;

    use super::*;

    #[test]
    fn simple_tool_description_serde() {
        let parameters = json!({
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
        });
        let desc = ToolDesc::new("temperature", "Get current temperature", parameters).unwrap();

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
