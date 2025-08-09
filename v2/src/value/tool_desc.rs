use std::{collections::HashMap, fmt};

use serde::{
    Deserialize, Deserializer, Serialize, Serializer,
    de::{self, Visitor},
    ser::SerializeStruct as _,
};

#[derive(Debug, Clone)]
pub enum JSONSchemaElement {
    String {
        description: Option<String>,
    },
    Number {
        description: Option<String>,
    },
    Boolean {
        description: Option<String>,
    },
    Object {
        properties: HashMap<String, Box<JSONSchemaElement>>,
        description: Option<String>,
    },
    Array {
        items: Option<Box<JSONSchemaElement>>,
        description: Option<String>,
    },
    Null,
}

impl Serialize for JSONSchemaElement {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("ToolDescriptionElement", 2)?;

        match self {
            JSONSchemaElement::String { description } => {
                state.serialize_field("type", "string")?;
                if let Some(description) = description {
                    state.serialize_field("description", description)?;
                }
            }
            JSONSchemaElement::Number { description } => {
                state.serialize_field("type", "number")?;
                if let Some(description) = description {
                    state.serialize_field("description", description)?;
                }
            }
            JSONSchemaElement::Boolean { description } => {
                state.serialize_field("type", "boolean")?;
                if let Some(description) = description {
                    state.serialize_field("description", description)?;
                }
            }
            JSONSchemaElement::Object {
                description,
                properties,
            } => {
                state.serialize_field("type", "object")?;
                if let Some(description) = description {
                    state.serialize_field("description", description)?;
                }
                state.serialize_field("properties", properties)?;
            }
            JSONSchemaElement::Array { description, items } => {
                state.serialize_field("type", "array")?;
                if let Some(description) = description {
                    state.serialize_field("description", description)?;
                }
                state.serialize_field("items", &items)?;
            }
            JSONSchemaElement::Null => {
                state.serialize_field("type", "null")?;
            }
        }

        state.end()
    }
}

impl<'de> Deserialize<'de> for JSONSchemaElement {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct JSONSchemaElementVisitor;

        impl<'de> Visitor<'de> for JSONSchemaElementVisitor {
            type Value = JSONSchemaElement;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a valid JSONSchemaElement")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::MapAccess<'de>,
            {
                let mut type_field = None;
                let mut description = None;
                let mut properties = None;
                let mut items = None;

                while let Some(key) = map.next_key::<String>()? {
                    match key.as_str() {
                        "type" => {
                            type_field = Some(map.next_value::<String>()?);
                        }
                        "description" => {
                            description = map.next_value()?;
                        }
                        "properties" => {
                            properties = map.next_value()?;
                        }
                        "items" => {
                            items = map.next_value()?;
                        }
                        _ => {}
                    }
                }

                match type_field.as_deref() {
                    Some("string") => Ok(JSONSchemaElement::String { description }),
                    Some("number") => Ok(JSONSchemaElement::Number { description }),
                    Some("boolean") => Ok(JSONSchemaElement::Boolean { description }),
                    Some("object") => Ok(JSONSchemaElement::Object {
                        description,
                        properties: properties.unwrap_or_default(),
                    }),
                    Some("array") => Ok(JSONSchemaElement::Array { description, items }),
                    Some("null") => Ok(JSONSchemaElement::Null),
                    _ => Err(de::Error::custom("Invalid type")),
                }
            }
        }

        deserializer.deserialize_map(JSONSchemaElementVisitor)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolDescription {
    pub name: String,

    pub description: String,

    #[serde(with = "tool_description_parameters_schema")]
    pub parameters: HashMap<String, JSONSchemaElement>,

    pub required: Vec<String>,

    #[serde(rename = "return")]
    pub ret: JSONSchemaElement,
}

impl ToolDescription {
    pub fn new(
        name: String,
        description: String,
        parameters: HashMap<String, JSONSchemaElement>,
        required: Vec<String>,
        ret: JSONSchemaElement,
    ) -> Self {
        Self {
            name,
            description,
            parameters,
            required,
            ret,
        }
    }
}

mod tool_description_parameters_schema {
    use super::*;
    use serde::ser::SerializeMap;

    pub fn serialize<S>(
        map: &HashMap<String, JSONSchemaElement>,
        serializer: S,
    ) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_map(Some(2))?;
        state.serialize_entry("type", "object")?;
        state.serialize_entry("properties", map)?;
        state.end()
    }

    pub fn deserialize<'de, D>(
        deserializer: D,
    ) -> Result<HashMap<String, JSONSchemaElement>, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Raw<T> {
            #[serde(rename = "type")]
            ty: Option<String>,

            properties: Option<HashMap<String, T>>,

            #[serde(flatten)]
            _extra: HashMap<String, serde::de::IgnoredAny>,
        }

        let raw = Raw::<JSONSchemaElement>::deserialize(deserializer)?;
        if let Some(t) = raw.ty.as_deref() {
            if t != "object" {
                return Err(<D::Error as de::Error>::custom(format!(
                    "parameters.type must be \"object\" (got {t})"
                )));
            }
        }
        Ok(raw.properties.unwrap_or_default())
    }
}
