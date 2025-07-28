use std::fmt;

use serde::{
    Deserialize, Deserializer, Serialize, Serializer,
    de::{self, MapAccess, Visitor},
    ser::SerializeMap as _,
};

// use super::ffi;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageDataType {
    Text,
    Image,
    Audio,
}

// impl From<MessageDataType> for ffi::DataType {
//     fn from(ty: MessageDataType) -> Self {
//         match ty {
//             MessageDataType::Text => Self::Text,
//             MessageDataType::Image => Self::Image,
//             MessageDataType::Audio => Self::Audio,
//         }
//     }
// }

#[derive(Clone, Debug)]
pub struct MessageContent {
    pub ty: MessageDataType,
    pub data_key: String,
    pub data_value: String,
}

// impl From<MessageContent> for ffi::Content {
//     fn from(content: MessageContent) -> Self {
//         ffi::Content {
//             ty: content.ty.into(),
//             data_key: content.data_key,
//             data_value: content.data_value,
//         }
//     }
// }

impl Serialize for MessageContent {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = serializer.serialize_map(Some(2))?;
        map.serialize_entry("type", &self.ty)?;
        map.serialize_entry(&self.data_key, &self.data_value)?;
        map.end()
    }
}

impl<'de> Deserialize<'de> for MessageContent {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_map(MessageContentVisitor)
    }
}

struct MessageContentVisitor;

impl<'de> Visitor<'de> for MessageContentVisitor {
    type Value = MessageContent;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str(r#"a map with "type" field and one other content key"#)
    }

    fn visit_map<M>(self, mut map: M) -> Result<MessageContent, M::Error>
    where
        M: MapAccess<'de>,
    {
        let mut ty: Option<MessageDataType> = None;
        let mut data_key: Option<String> = None;
        let mut data_value: Option<String> = None;

        while let Some(key) = map.next_key::<String>()? {
            if key == "type" {
                if ty.is_some() {
                    return Err(de::Error::duplicate_field("type"));
                }
                ty = Some(map.next_value()?);
            } else {
                if data_key.is_some() {
                    return Err(de::Error::custom("multiple content fields found"));
                }
                data_key = Some(key);
                data_value = Some(map.next_value()?);
            }
        }

        let ty = ty.ok_or_else(|| de::Error::missing_field("type"))?;
        let data_key = data_key.ok_or_else(|| de::Error::custom("missing content field"))?;
        let data_value = data_value.ok_or_else(|| de::Error::custom("missing content value"))?;

        Ok(MessageContent {
            ty,
            data_key,
            data_value,
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RoleType {
    System,
    User,
    Assistant,
    Tool,
}

// impl From<RoleType> for ffi::RoleType {
//     fn from(ty: RoleType) -> Self {
//         match ty {
//             RoleType::System => Self::System,
//             RoleType::User => Self::User,
//             RoleType::Assistant => Self::Assistant,
//             RoleType::Tool => Self::Tool,
//         }
//     }
// }

pub struct Message {
    role: RoleType,
    content_key: String,
    content_value: Vec<MessageContent>,
}

impl Serialize for Message {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = serializer.serialize_map(Some(2))?;
        map.serialize_entry("role", &self.role)?;
        map.serialize_entry(&self.content_key, &self.content_value)?;
        map.end()
    }
}

impl<'de> Deserialize<'de> for Message {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_map(MessageVisitor)
    }
}

struct MessageVisitor;

impl<'de> Visitor<'de> for MessageVisitor {
    type Value = Message;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str(r#"a map with "type" field and one other content key"#)
    }

    fn visit_map<M>(self, mut map: M) -> Result<Message, M::Error>
    where
        M: MapAccess<'de>,
    {
        let mut role: Option<RoleType> = None;
        let mut content_key: Option<String> = None;
        let mut content_value: Option<Vec<MessageContent>> = None;

        while let Some(key) = map.next_key::<String>()? {
            if key == "type" {
                if role.is_some() {
                    return Err(de::Error::duplicate_field("type"));
                }
                role = Some(map.next_value()?);
            } else {
                if content_key.is_some() {
                    return Err(de::Error::custom("multiple content fields found"));
                }
                content_key = Some(key);
                content_value = Some(map.next_value()?);
            }
        }

        let role = role.ok_or_else(|| de::Error::missing_field("type"))?;
        let content_key = content_key.ok_or_else(|| de::Error::custom("missing content field"))?;
        let content_value =
            content_value.ok_or_else(|| de::Error::custom("missing content value"))?;

        Ok(Message {
            role,
            content_key,
            content_value,
        })
    }
}
