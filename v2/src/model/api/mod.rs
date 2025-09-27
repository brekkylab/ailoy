use crate::model::ThinkEffort;

pub mod anthropic;
pub mod chat_completion;
pub mod gemini;
pub mod openai;
pub mod sse;
// pub mod xai;

#[derive(Clone, Debug, PartialEq)]
struct RequestInfo {
    pub model: Option<String>,

    pub system_message: Option<String>,

    pub stream: bool,

    pub think_effort: ThinkEffort,

    pub temperature: Option<f64>,

    pub top_p: Option<f64>,

    pub max_tokens: Option<i32>,
}

// impl LMConfig {
//     pub fn with_model(mut self, model: impl Into<String>) -> Self {
//         self.model = Some(model.into());
//         self
//     }
// }

// #[derive(Default, Debug)]
// pub struct LMConfigBuilder {
//     system_message: Option<String>,
//     stream: bool,
//     think_effort: ThinkEffort,
//     temperature: Option<f64>,
//     top_p: Option<f64>,
//     max_tokens: Option<i32>,
//     // output_schema: Grammar,
// }

// impl LMConfigBuilder {
//     pub fn new() -> Self {
//         Self::default()
//     }

//     pub fn system_message(mut self, system_message: impl Into<String>) -> Self {
//         self.system_message = Some(system_message.into());
//         self
//     }

//     pub fn stream(mut self, stream: bool) -> Self {
//         self.stream = stream;
//         self
//     }

//     pub fn thinking_option(mut self, thinking_option: ThinkingOption) -> Self {
//         self.thinking_option = thinking_option;
//         self
//     }

//     pub fn temperature(mut self, temperature: f64) -> Self {
//         self.temperature = Some(temperature);
//         self
//     }

//     pub fn top_p(mut self, top_p: f64) -> Self {
//         self.top_p = Some(top_p);
//         self
//     }

//     pub fn max_tokens(mut self, max_tokens: i32) -> Self {
//         self.max_tokens = Some(max_tokens);
//         self
//     }

//     // pub fn grammar(mut self, grammar: Grammar) -> Self {
//     //     self.grammar = grammar;
//     //     self
//     // }

//     pub fn build(self) -> LMConfig {
//         LMConfig {
//             system_message: self.system_message,
//             stream: self.stream,
//             thinking_option: self.thinking_option,
//             temperature: self.temperature,
//             top_p: self.top_p,
//             max_tokens: self.max_tokens,
//             // output_schema: self.output_schema,
//             model: None,
//         }
//     }
// }

// impl Serialize for LMConfig {
//     fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
//     where
//         S: Serializer,
//     {
//         let mut field_count = 3;
//         if self.system_message.is_some() {
//             field_count += 1;
//         }
//         if self.temperature.is_some() {
//             field_count += 1;
//         }
//         if self.top_p.is_some() {
//             field_count += 1;
//         }
//         if self.max_tokens.is_some() {
//             field_count += 1;
//         }

//         let mut state = serializer.serialize_struct("Config", field_count)?;

//         if let Some(ref val) = self.system_message {
//             state.serialize_field("system_message", val)?;
//         }
//         state.serialize_field("stream", &self.stream)?;
//         state.serialize_field("thinking_option", &self.thinking_option)?;

//         if let Some(ref val) = self.temperature {
//             state.serialize_field("temperature", val)?;
//         }
//         if let Some(ref val) = self.top_p {
//             state.serialize_field("top_p", val)?;
//         }

//         if let Some(ref val) = self.max_tokens {
//             state.serialize_field("max_tokens", val)?;
//         }
//         // state.serialize_field("output_schema", &self.output_schema)?;

//         state.end()
//     }
// }

// impl<'de> Deserialize<'de> for LMConfig {
//     fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
//     where
//         D: serde::Deserializer<'de>,
//     {
//         todo!()
//     }
// }
