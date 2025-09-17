use napi::{Env, Error as NapiError, Result, Status, bindgen_prelude::*};
use napi_derive::napi;

use crate::{
    ffi::node::common::{get_property, json_parse, json_stringify},
    value::{FinishReason, Message, MessageOutput, Part, Role},
};

////////////
/// Part ///
////////////

#[napi(js_name = "Part")]
pub struct JsPart {
    inner: Part,
}

impl Into<Part> for JsPart {
    fn into(self) -> Part {
        self.inner
    }
}

impl From<Part> for JsPart {
    fn from(part: Part) -> Self {
        Self { inner: part }
    }
}

impl FromNapiValue for JsPart {
    unsafe fn from_napi_value(env: sys::napi_env, napi_val: sys::napi_value) -> Result<Self> {
        let env = Env::from(env);
        let obj = unsafe { Object::from_napi_value(env.raw(), napi_val)? };

        let part_type: String = get_property(obj, "partType")?;
        match part_type.as_str() {
            "Text" => {
                let text = get_property(obj, "text")?;
                Ok(Part::Text(text).into())
            }
            "Function" => {
                let id: String = get_property(obj, "id")?;
                let name: String = get_property(obj, "name")?;
                let arguments: Object = get_property(obj, "arguments")?;
                Ok(Part::Function {
                    id,
                    name,
                    arguments: json_stringify(env, arguments)?,
                }
                .into())
            }
            "ImageURL" => {
                let url: String = get_property(obj, "url")?;
                Ok(Part::ImageURL(url).into())
            }
            "ImageData" => {
                let data: String = get_property(obj, "data")?;
                let mime_type: String = get_property(obj, "mimeType")?;
                Ok(Part::ImageData { data, mime_type }.into())
            }
            _ => Err(NapiError::new(
                Status::Unknown,
                format!("Unknown partType: {}", part_type),
            )),
        }
    }
}

#[napi]
impl JsPart {
    #[napi(factory)]
    pub fn new_text(text: String) -> Self {
        Self {
            inner: Part::new_text(text),
        }
    }

    #[napi(factory)]
    pub fn new_function(env: Env, id: String, name: String, arguments: Object) -> Self {
        let arguments_str = json_stringify(env, arguments).unwrap();
        Self {
            inner: Part::new_function(id, name, arguments_str),
        }
    }

    #[napi(factory)]
    pub fn new_image_url(url: String) -> Self {
        Self {
            inner: Part::new_image_url(url),
        }
    }

    #[napi(factory)]
    pub fn new_image_data(data: String, mime_type: String) -> Self {
        Self {
            inner: Part::new_image_data(data, mime_type),
        }
    }

    #[napi(getter)]
    pub fn part_type(&self) -> Result<String> {
        Ok(self.inner.as_ref().to_string())
    }

    #[napi(getter)]
    pub fn text(&self) -> Result<Option<String>> {
        Ok(match &self.inner {
            Part::Text(text) => Some(text.clone()),
            _ => None,
        })
    }

    #[napi(setter)]
    pub fn set_text(&mut self, text: String) -> Result<()> {
        match &mut self.inner {
            Part::Text(text_) => {
                *text_ = text;
                Ok(())
            }
            _ => Err(Error::new(
                Status::InvalidArg,
                "This part is not a type of Text",
            )),
        }
    }

    #[napi(getter)]
    pub fn id(&self) -> Result<Option<String>> {
        Ok(match &self.inner {
            Part::Function { id, .. } => Some(id.clone()),
            _ => None,
        })
    }

    #[napi(setter)]
    pub fn set_id(&mut self, id: String) -> Result<()> {
        match &mut self.inner {
            Part::Function { id: id_, .. } => {
                *id_ = id;
                Ok(())
            }
            _ => Err(Error::new(
                Status::InvalidArg,
                "This part is not a type of Function",
            )),
        }
    }

    #[napi(getter)]
    pub fn name(&self) -> Result<Option<String>> {
        Ok(match &self.inner {
            Part::Function { name, .. } => Some(name.clone()),
            _ => None,
        })
    }

    #[napi(setter)]
    pub fn set_name(&mut self, name: String) -> Result<()> {
        match &mut self.inner {
            Part::Function { name: name_, .. } => {
                *name_ = name;
                Ok(())
            }
            _ => Err(Error::new(
                Status::InvalidArg,
                "This part is not a type of Function",
            )),
        }
    }

    #[napi(getter)]
    pub fn arguments<'a>(&'a self, env: Env) -> Result<Option<Object<'a>>> {
        Ok(match &self.inner {
            Part::Function { arguments, .. } => Some(json_parse(env, arguments.clone())?),
            _ => None,
        })
    }

    #[napi(setter)]
    pub fn set_arguments(&mut self, env: Env, arguments: Object) -> Result<()> {
        match &mut self.inner {
            Part::Function {
                arguments: arguments_,
                ..
            } => {
                *arguments_ = json_stringify(env, arguments)?;
                Ok(())
            }
            _ => Err(Error::new(
                Status::InvalidArg,
                "This part is not a type of Function",
            )),
        }
    }

    #[napi(getter)]
    pub fn url(&self) -> Result<Option<String>> {
        Ok(match &self.inner {
            Part::ImageURL(url) => Some(url.clone()),
            _ => None,
        })
    }

    #[napi(setter)]
    pub fn set_url(&mut self, url: String) -> Result<()> {
        match &mut self.inner {
            Part::ImageURL(url_) => {
                *url_ = url;
                Ok(())
            }
            _ => Err(Error::new(
                Status::InvalidArg,
                "This part is not a type of ImageURL",
            )),
        }
    }

    #[napi(getter)]
    pub fn data(&self) -> Result<Option<String>> {
        Ok(match &self.inner {
            Part::ImageData { data, .. } => Some(data.clone()),
            _ => None,
        })
    }

    #[napi(setter)]
    pub fn set_data(&mut self, data: String) -> Result<()> {
        match &mut self.inner {
            Part::ImageData { data: data_, .. } => {
                *data_ = data;
                Ok(())
            }
            _ => Err(Error::new(
                Status::InvalidArg,
                "This part is not a type of ImageData",
            )),
        }
    }

    #[napi(getter)]
    pub fn mime_type(&self) -> Result<Option<String>> {
        Ok(match &self.inner {
            Part::ImageData { mime_type, .. } => Some(mime_type.clone()),
            _ => None,
        })
    }

    #[napi(setter)]
    pub fn set_mime_type(&mut self, mime_type: String) -> Result<()> {
        match &mut self.inner {
            Part::ImageData {
                mime_type: mime_type_,
                ..
            } => {
                *mime_type_ = mime_type;
                Ok(())
            }
            _ => Err(Error::new(
                Status::InvalidArg,
                "This part is not a type of ImageData",
            )),
        }
    }

    #[napi(js_name = "toJSON")]
    pub fn to_json(&'_ self, env: Env) -> Result<Object<'_>> {
        let mut obj = Object::new(&env)?;
        obj.set("partType", self.part_type())?;
        match self.inner {
            Part::Text(..) => {
                obj.set("text", self.text())?;
            }
            Part::Function { .. } => {
                obj.set("id", self.id())?;
                obj.set("name", self.name())?;
                obj.set("arguments", self.arguments(env))?;
            }
            Part::ImageURL(..) => {
                obj.set("url", self.url())?;
            }
            Part::ImageData { .. } => {
                obj.set("data", self.data())?;
                obj.set("mimeType", self.mime_type())?;
            }
            _ => {
                return Err(Error::new(Status::InvalidArg, "Unsupported part type"));
            }
        }
        Ok(obj)
    }

    #[napi]
    pub fn to_string(&'_ self, env: Env) -> Result<String> {
        let json = self.to_json(env)?;
        let str = json_stringify(env, json)?;
        Ok(str)
    }
}

///////////////
/// Message ///
///////////////

#[napi(js_name = "Message")]
pub struct JsMessage {
    inner: Message,
}

impl Into<Message> for JsMessage {
    fn into(self) -> Message {
        self.inner
    }
}

impl From<Message> for JsMessage {
    fn from(msg: Message) -> Self {
        Self { inner: msg }
    }
}

impl FromNapiValue for JsMessage {
    unsafe fn from_napi_value(env: sys::napi_env, napi_val: sys::napi_value) -> Result<Self> {
        let env = Env::from(env);
        let obj = unsafe { Object::from_napi_value(env.raw(), napi_val)? };

        let role: Option<Role> = obj.get("role")?;

        let contents_array: Array = get_property(obj, "contents")?;
        let contents_len = contents_array.get_array_length()?;
        let mut contents = Vec::with_capacity(contents_len as usize);
        for i in 0..contents_len {
            let part: JsPart = contents_array.get(i)?.ok_or_else(|| {
                Error::new(Status::InvalidArg, format!("Missing contents[{}]", i))
            })?;
            contents.push(part.into());
        }

        let tool_calls_array: Array = get_property(obj, "toolCalls")?;
        let tool_calls_len = tool_calls_array.get_array_length()?;
        let mut tool_calls = Vec::with_capacity(tool_calls_len as usize);
        for i in 0..tool_calls_len {
            let part: JsPart = tool_calls_array.get(i)?.ok_or_else(|| {
                Error::new(Status::InvalidArg, format!("Missing toolCalls[{}]", i))
            })?;
            tool_calls.push(part.into());
        }

        let reasoning: String = get_property(obj, "reasoning")?;

        let tool_call_id: Option<String> = get_property(obj, "toolCallId")?;

        Ok(Message {
            role,
            contents,
            reasoning,
            tool_calls,
            tool_call_id,
        }
        .into())
    }
}

#[napi]
impl JsMessage {
    #[napi(constructor)]
    pub fn new() -> Self {
        Self {
            inner: Message::new(),
        }
    }

    #[napi(getter)]
    pub fn role(&self) -> Option<Role> {
        self.inner.role.clone()
    }

    #[napi(setter)]
    pub fn set_role(&mut self, role: Role) {
        self.inner.role = Some(role)
    }

    #[napi(getter)]
    pub fn contents(&self) -> Vec<JsPart> {
        self.inner
            .contents
            .clone()
            .into_iter()
            .map(|c| c.into())
            .collect()
    }

    #[napi(setter)]
    pub fn set_contents(&mut self, contents: Vec<JsPart>) {
        self.inner.contents = contents.into_iter().map(|c| c.into()).collect();
    }

    #[napi(getter)]
    pub fn reasoning(&self) -> String {
        self.inner.reasoning.clone()
    }

    #[napi(setter)]
    pub fn set_reasoning(&mut self, reasoning: String) {
        self.inner.reasoning = reasoning;
    }

    #[napi(getter)]
    pub fn tool_calls(&self) -> Vec<JsPart> {
        self.inner
            .tool_calls
            .clone()
            .into_iter()
            .map(|c| c.into())
            .collect()
    }

    #[napi(setter)]
    pub fn set_tool_calls(&mut self, tool_calls: Vec<JsPart>) {
        self.inner.tool_calls = tool_calls.into_iter().map(|c| c.into()).collect();
    }

    #[napi(getter)]
    pub fn tool_call_id(&self) -> Option<String> {
        self.inner.tool_call_id.clone()
    }

    #[napi(setter)]
    pub fn set_tool_call_id(&mut self, tool_call_id: String) {
        self.inner.tool_call_id = Some(tool_call_id);
    }

    #[napi(js_name = "toJSON")]
    pub fn to_json(&'_ self, env: Env) -> Result<Object<'_>> {
        let mut obj = Object::new(&env)?;
        obj.set("role", self.role())?;
        obj.set("contents", self.contents())?;
        obj.set("reasoning", self.reasoning())?;
        obj.set("toolCalls", self.tool_calls())?;
        obj.set("toolCallId", self.tool_call_id())?;
        Ok(obj)
    }

    #[napi]
    pub fn to_string(&self, env: Env) -> Result<String> {
        let obj = self.to_json(env)?;
        let str = json_stringify(env, obj)?;
        Ok(str)
    }
}

/////////////////////
/// MessageOutput ///
/////////////////////

#[napi(js_name = "MessageOutput")]
pub struct JsMessageOutput {
    inner: MessageOutput,
}

impl Into<MessageOutput> for JsMessageOutput {
    fn into(self) -> MessageOutput {
        self.inner
    }
}

impl From<MessageOutput> for JsMessageOutput {
    fn from(msg: MessageOutput) -> Self {
        Self { inner: msg }
    }
}

impl FromNapiValue for JsMessageOutput {
    unsafe fn from_napi_value(env: sys::napi_env, napi_val: sys::napi_value) -> Result<Self> {
        let env = Env::from(env);
        let obj = unsafe { Object::from_napi_value(env.raw(), napi_val)? };

        let delta: JsMessage = get_property(obj, "delta")?;
        let finish_reason: Option<FinishReason> = obj.get("finishReason")?;

        Ok(MessageOutput {
            delta: delta.into(),
            finish_reason,
        }
        .into())
    }
}

#[napi]
impl JsMessageOutput {
    #[napi(getter)]
    pub fn delta(&self) -> JsMessage {
        self.inner.delta.clone().into()
    }

    #[napi(getter)]
    pub fn finish_reason(&self) -> Option<FinishReason> {
        self.inner.finish_reason.clone()
    }

    #[napi(js_name = "toJSON")]
    pub fn to_json(&'_ self, env: Env) -> Result<Object<'_>> {
        let mut obj = Object::new(&env)?;
        obj.set("delta", self.delta())?;
        obj.set("finishReason", self.finish_reason())?;
        Ok(obj)
    }

    #[napi]
    pub fn to_string(&self, env: Env) -> Result<String> {
        let obj = self.to_json(env)?;
        let str = json_stringify(env, obj)?;
        Ok(str)
    }
}
