use std::fmt;

use serde::{
    Deserialize, Deserializer, Serialize, Serializer,
    de::{self, MapAccess, Visitor},
    ser::SerializeMap as _,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageDataType {
    Text,
    Image,
    Audio,
}

#[derive(Clone, Debug)]
pub struct MessageContent {
    pub ty: MessageDataType,
    pub key: String,
    pub value: String,
}

impl Serialize for MessageContent {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = serializer.serialize_map(Some(2))?;
        map.serialize_entry("type", &self.ty)?;
        map.serialize_entry(&self.key, &self.value)?;
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
        let mut key: Option<String> = None;
        let mut value: Option<String> = None;

        while let Some(k) = map.next_key::<String>()? {
            if k == "type" {
                if ty.is_some() {
                    return Err(de::Error::duplicate_field("type"));
                }
                ty = Some(map.next_value()?);
            } else {
                if key.is_some() {
                    return Err(de::Error::custom("multiple data fields found"));
                }
                key = Some(k);
                value = Some(map.next_value()?);
            }
        }

        let ty = ty.ok_or_else(|| de::Error::missing_field("type"))?;
        let key = key.ok_or_else(|| de::Error::custom("missing data field"))?;
        let value = value.ok_or_else(|| de::Error::custom("missing data value"))?;

        Ok(MessageContent { ty, key, value })
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

#[derive(Clone, Debug)]
pub struct Message {
    role: RoleType,
    key: String,
    value: Vec<MessageContent>,
}

impl Serialize for Message {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = serializer.serialize_map(Some(2))?;
        map.serialize_entry("role", &self.role)?;
        map.serialize_entry(&self.key, &self.value)?;
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
        let mut key: Option<String> = None;
        let mut value: Option<Vec<MessageContent>> = None;

        while let Some(k) = map.next_key::<String>()? {
            if k == "role" {
                if role.is_some() {
                    return Err(de::Error::duplicate_field("role"));
                }
                role = Some(map.next_value()?);
            } else {
                if key.is_some() {
                    return Err(de::Error::custom("multiple content fields found"));
                }
                key = Some(k);
                value = Some(map.next_value()?);
            }
        }

        let role = role.ok_or_else(|| de::Error::missing_field("role"))?;
        let key = key.ok_or_else(|| de::Error::custom("missing content field"))?;
        let value = value.ok_or_else(|| de::Error::custom("missing content value"))?;

        Ok(Message { role, key, value })
    }
}
