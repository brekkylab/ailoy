use std::sync::OnceLock;

use futures::StreamExt;
use napi::{Error, Result as NapiResult, Status, bindgen_prelude::*};
use napi_derive::napi;
use tokio::sync::mpsc;

use crate::{
    ffi::node::value::{Message, MessageOutput},
    model::{LanguageModel, openai::OpenAILanguageModel},
    value::{Message as _Message, MessageOutput as _MessageOutput},
};

static RUNTIME: OnceLock<tokio::runtime::Runtime> = OnceLock::new();

fn get_or_create_runtime() -> &'static tokio::runtime::Runtime {
    RUNTIME.get_or_init(|| tokio::runtime::Runtime::new().expect("Failed to create tokio runtime"))
}

#[napi(object)]
pub struct LanguageModelIteratorResult {
    pub value: Option<MessageOutput>,
    pub done: bool,
}

#[napi]
pub struct LanguageModelRunIterator {
    rx: mpsc::UnboundedReceiver<std::result::Result<_MessageOutput, String>>,
}

#[napi]
impl LanguageModelRunIterator {
    #[napi]
    pub async unsafe fn next(&mut self) -> Result<LanguageModelIteratorResult> {
        match self.rx.recv().await {
            Some(Ok(output)) => Ok(LanguageModelIteratorResult {
                value: Some(output.into()),
                done: false,
            }),
            Some(Err(e)) => Err(Error::new(Status::GenericFailure, e)),
            None => Ok(LanguageModelIteratorResult {
                value: None,
                done: true,
            }),
        }
    }
}

pub trait LanguageModelMethods<T: LanguageModel + 'static> {
    fn get_inner(&self) -> Result<T>;

    fn create_iterator(&self, messages: Vec<_Message>) -> NapiResult<LanguageModelRunIterator> {
        let model = self.get_inner()?;
        let (tx, rx) = mpsc::unbounded_channel();

        let rt = get_or_create_runtime();

        rt.spawn(async move {
            let mut model = model;
            let mut stream = model.run(messages, vec![]).boxed();

            while let Some(item) = stream.next().await {
                if tx.send(item).is_err() {
                    break;
                }
                tokio::task::yield_now().await;
            }
        });

        Ok(LanguageModelRunIterator { rx })
    }
}

#[napi(js_name = "OpenAILanguageModel")]
pub struct JSOpenAILanguageModel {
    inner: OpenAILanguageModel,
}

impl LanguageModelMethods<OpenAILanguageModel> for JSOpenAILanguageModel {
    fn get_inner(&self) -> Result<OpenAILanguageModel> {
        Ok(self.inner.clone())
    }
}

#[napi]
impl JSOpenAILanguageModel {
    #[napi(constructor)]
    pub fn new(model_name: String, api_key: String) -> Result<Self> {
        Ok(Self {
            inner: OpenAILanguageModel::new(model_name, api_key),
        })
    }

    #[napi]
    pub fn iterator(&self, messages: Vec<Message>) -> Result<LanguageModelRunIterator> {
        self.create_iterator(messages.into_iter().map(|msg| msg.into()).collect())
    }
}
