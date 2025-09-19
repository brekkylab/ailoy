use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "python", pyo3_stub_gen_derive::gen_stub_pyclass)]
#[cfg_attr(feature = "python", pyo3::pyclass)]
#[cfg_attr(feature = "nodejs", napi_derive::napi(object))]
pub struct ToolDesc {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
    pub returns: Option<serde_json::Value>,
}

impl ToolDesc {
    pub fn new(
        name: String,
        description: String,
        parameters: serde_json::Value,
        returns: Option<serde_json::Value>,
    ) -> Result<Self, String> {
        jsonschema::validator_for(&parameters)
            .map_err(|e| format!("parameters is not a valid jsonschema: {}", e.to_string()))?;
        if let Some(returns) = &returns {
            jsonschema::validator_for(&returns)
                .map_err(|e| format!("returns is not a valid jsonschema: {}", e.to_string()))?;
        }
        Ok(Self {
            name,
            description,
            parameters,
            returns,
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
        let returns = json!({
            "type": "number"
        });
        let desc = ToolDesc::new(
            "temperature".into(),
            "Get current temperature".into(),
            parameters,
            Some(returns),
        )
        .unwrap();

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
