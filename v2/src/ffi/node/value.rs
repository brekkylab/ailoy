use std::str::FromStr;

use napi::{Error, Result, Status};
use napi_derive::napi;

use crate::value::{Message, MessageDelta, Part, Role};

fn type_err(msg: impl Into<String>) -> Error {
    Error::new(Status::InvalidArg, msg.into())
}

fn value_err(msg: impl Into<String>) -> Error {
    Error::new(Status::InvalidArg, msg.into())
}

#[napi(object)]
pub struct PartInitializer {
    #[napi(js_name = "type")]
    pub type_: String,

    pub id: Option<String>,

    pub text: Option<String>,

    pub url: Option<String>,

    pub data: Option<String>,

    #[napi(js_name = "function")]
    pub function_: Option<String>,
}

#[napi]
pub struct NodePart {
    inner: Part,
}

impl NodePart {
    fn from_inner(inner: Part) -> Self {
        Self { inner }
    }
}

#[napi]
impl NodePart {
    #[napi(factory)]
    pub fn new(init: PartInitializer) -> Result<Self> {
        let inner = match init.type_.as_str() {
            "text" => Part::Text(
                init.text
                    .ok_or_else(|| type_err("text= required for type='text'"))?,
            ),
            "function" => Part::Function {
                id: init.id,
                function: init
                    .function_
                    .ok_or_else(|| type_err("function= required for type='function'"))?,
            },
            "image" => {
                if let Some(u) = init.url {
                    let parsed = url::Url::parse(&u).map_err(|e| value_err(e.to_string()))?;
                    Part::ImageURL(parsed)
                } else if let Some(b) = init.data {
                    Part::ImageData(b)
                } else {
                    return Err(type_err("image needs url= or data="));
                }
            }
            other => return Err(value_err(format!("unknown type: {other}"))),
        };
        Ok(Self { inner })
    }

    #[napi(getter, js_name = "type")]
    pub fn type_(&self) -> String {
        match &self.inner {
            Part::Text(_) => "text",
            Part::Function { .. } => "function",
            Part::ImageURL(_) | Part::ImageData(_) => "image",
            Part::Audio { .. } => "audio",
        }
        .to_string()
    }

    #[napi(getter)]
    pub fn text(&self) -> Option<String> {
        self.inner.get_text().map(|s| s.to_string())
    }

    #[napi(getter)]
    pub fn id(&self) -> Option<String> {
        self.inner.get_function_id()
    }

    #[napi(getter, js_name = "function")]
    pub fn function_(&self) -> Option<String> {
        self.inner.get_function().cloned()
    }

    #[napi(getter)]
    pub fn url(&self) -> Option<String> {
        match &self.inner {
            Part::ImageURL(u) => Some(u.as_str().to_string()),
            _ => None,
        }
    }

    #[napi(getter)]
    pub fn data(&self) -> Option<String> {
        match &self.inner {
            Part::ImageData(b) => Some(b.clone()),
            _ => None,
        }
    }

    #[napi(factory, js_name = "fromJSON")]
    pub fn from_json(s: String) -> Result<Self> {
        let inner: Part = serde_json::from_str(&s).map_err(|e| value_err(e.to_string()))?;
        Ok(Self { inner })
    }

    #[napi(js_name = "toJSON")]
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string(&self.inner).map_err(|e| value_err(e.to_string()))
    }

    #[napi(js_name = "toString")]
    pub fn to_string_js(&self) -> String {
        self.inner.to_string()
    }
}

// =====================================
// MessageDelta
// =====================================

#[napi]
pub struct NodeMessageDelta {
    inner: MessageDelta,
}

#[napi]
impl NodeMessageDelta {
    #[napi(factory, js_name = "fromJSON")]
    pub fn from_json(s: String) -> Result<Self> {
        let inner: MessageDelta = serde_json::from_str(&s).map_err(|e| value_err(e.to_string()))?;
        Ok(Self { inner })
    }

    #[napi(js_name = "toJSON")]
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string(&self.inner).map_err(|e| value_err(e.to_string()))
    }

    #[napi(js_name = "toString")]
    pub fn to_string_js(&self) -> String {
        self.inner.to_string()
    }
}

#[napi]
pub struct NodeMessage {
    inner: Message,
}

#[napi]
impl NodeMessage {
    /// new(role: string)  // role in {"system","user","assistant","tool"}
    #[napi(constructor)]
    pub fn new(role: String) -> Result<Self> {
        let role = Role::from_str(&role).map_err(|_| value_err(role))?;
        Ok(Self {
            inner: Message::new(role),
        })
    }

    #[napi(getter)]
    pub fn role(&self) -> String {
        self.inner.role.to_string()
    }

    // ---------- content ----------
    #[napi(getter)]
    pub fn content(&self) -> Vec<NodePart> {
        self.inner
            .content
            .clone()
            .into_iter()
            .map(NodePart::from_inner)
            .collect()
    }

    #[napi(setter)]
    pub fn set_content(&mut self, parts: Vec<&NodePart>) {
        self.inner.content = parts.into_iter().map(|p| p.inner.clone()).collect();
    }

    #[napi(js_name = "appendContent")]
    pub fn append_content(&mut self, part: &NodePart) {
        self.inner.content.push(part.inner.clone());
    }

    // ---------- reasoning ----------
    #[napi(getter)]
    pub fn reasoning(&self) -> Vec<NodePart> {
        self.inner
            .reasoning
            .clone()
            .into_iter()
            .map(NodePart::from_inner)
            .collect()
    }

    #[napi(setter)]
    pub fn set_reasoning(&mut self, parts: Vec<&NodePart>) {
        self.inner.reasoning = parts.into_iter().map(|p| p.inner.clone()).collect();
    }

    #[napi(js_name = "appendReasoning")]
    pub fn append_reasoning(&mut self, part: &NodePart) {
        self.inner.reasoning.push(part.inner.clone());
    }

    // ---------- tool_calls ----------
    #[napi(getter, js_name = "tool_calls")]
    pub fn tool_calls(&self) -> Vec<NodePart> {
        self.inner
            .tool_calls
            .clone()
            .into_iter()
            .map(NodePart::from_inner)
            .collect()
    }

    #[napi(setter, js_name = "tool_calls")]
    pub fn set_tool_calls(&mut self, parts: Vec<&NodePart>) {
        self.inner.tool_calls = parts.into_iter().map(|p| p.inner.clone()).collect();
    }

    #[napi(js_name = "appendToolCall")]
    pub fn append_tool_call(&mut self, part: &NodePart) {
        self.inner.tool_calls.push(part.inner.clone());
    }

    // ---------- JSON / String ----------
    #[napi(factory, js_name = "fromJSON")]
    pub fn from_json(s: String) -> Result<Self> {
        let inner: Message = serde_json::from_str(&s).map_err(|e| value_err(e.to_string()))?;
        Ok(Self { inner })
    }

    #[napi(js_name = "toJSON")]
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string(&self.inner).map_err(|e| value_err(e.to_string()))
    }

    #[napi(js_name = "toString")]
    pub fn to_string_js(&self) -> String {
        self.inner.to_string()
    }
}
