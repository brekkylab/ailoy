use serde::{
    Deserialize, Serialize,
    ser::{SerializeStruct, Serializer},
};

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub enum ReasoningOption {
    #[default]
    Disable,
    Enable,
    Low,
    Medium,
    High,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub enum OutputSchema {
    #[default]
    Text,
    JSON,
    JSONSchema(String),
    Regex(String),
    CFG(String),
}

#[derive(Clone, Debug, PartialEq)]
// #[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(
    feature = "python",
    pyo3_stub_gen_derive::gen_stub_pyclass_complex_enum
)]
#[cfg_attr(feature = "python", pyo3::pyclass(eq))]
pub struct Config {
    pub model: Option<String>,
    pub system_message: Option<String>,
    pub stream: bool,
    pub reasoning_option: ReasoningOption,

    pub temperature: Option<f64>,
    pub top_p: Option<f64>,

    pub max_tokens: Option<i32>,
    // pub output_schema: OutputSchema,
}

impl Config {
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }
}

#[derive(Default, Debug)]
pub struct ConfigBuilder {
    system_message: Option<String>,
    stream: bool,
    reasoning_option: ReasoningOption,
    temperature: Option<f64>,
    top_p: Option<f64>,
    max_tokens: Option<i32>,
    // output_schema: OutputSchema,
}

impl ConfigBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn system_message(mut self, system_message: impl Into<String>) -> Self {
        self.system_message = Some(system_message.into());
        self
    }

    pub fn stream(mut self, stream: bool) -> Self {
        self.stream = stream;
        self
    }

    pub fn reasoning_option(mut self, reasoning_option: ReasoningOption) -> Self {
        self.reasoning_option = reasoning_option;
        self
    }

    pub fn temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn top_p(mut self, top_p: f64) -> Self {
        self.top_p = Some(top_p);
        self
    }

    pub fn max_tokens(mut self, max_tokens: i32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    // pub fn output_schema(mut self, output_schema: OutputSchema) -> Self {
    //     self.output_schema = output_schema;
    //     self
    // }

    pub fn build(self) -> Config {
        Config {
            system_message: self.system_message,
            stream: self.stream,
            reasoning_option: self.reasoning_option,
            temperature: self.temperature,
            top_p: self.top_p,
            max_tokens: self.max_tokens,
            // output_schema: self.output_schema,
            model: None,
        }
    }
}

impl Serialize for Config {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut field_count = 3;
        if self.system_message.is_some() {
            field_count += 1;
        }
        if self.temperature.is_some() {
            field_count += 1;
        }
        if self.top_p.is_some() {
            field_count += 1;
        }
        if self.max_tokens.is_some() {
            field_count += 1;
        }

        let mut state = serializer.serialize_struct("Config", field_count)?;

        if let Some(ref val) = self.system_message {
            state.serialize_field("system_message", val)?;
        }
        state.serialize_field("stream", &self.stream)?;
        state.serialize_field("reasoning_option", &self.reasoning_option)?;

        if let Some(ref val) = self.temperature {
            state.serialize_field("temperature", val)?;
        }
        if let Some(ref val) = self.top_p {
            state.serialize_field("top_p", val)?;
        }

        if let Some(ref val) = self.max_tokens {
            state.serialize_field("max_tokens", val)?;
        }
        // state.serialize_field("output_schema", &self.output_schema)?;

        state.end()
    }
}

impl<'de> Deserialize<'de> for Config {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        todo!()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length, // max_output_tokens
    ToolCall,
    Refusal(String), // content_filter, refusal
}
