use std::{any::TypeId, sync::Arc};

use downcast_rs::{Downcast, impl_downcast};
use futures::lock::Mutex;
use serde::Serialize;

use crate::{
    utils::{BoxStream, MaybeSend, MaybeSync},
    value::{Message, MessageOutput, ToolDesc},
};

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub enum ThinkEffort {
    #[default]
    Disable,
    Enable,
    Low,
    Medium,
    High,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub enum Grammar {
    #[default]
    Plain,
    JSON,
    JSONSchema(String),
    Regex(String),
    CFG(String),
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct InferenceConfig {
    pub think_effort: Option<ThinkEffort>,

    pub temperature: Option<ordered_float::OrderedFloat<f64>>,

    pub top_p: Option<ordered_float::OrderedFloat<f64>>,

    pub max_tokens: Option<i32>,

    pub grammar: Option<Grammar>,
}

pub trait LanguageModel: Downcast + MaybeSend + MaybeSync {
    /// Runs the language model with the given tools and messages, returning a stream of `MessageOutput`s.
    fn run<'a>(
        &'a mut self,
        msg: Vec<Message>,
        tools: Vec<ToolDesc>,
        config: InferenceConfig,
    ) -> BoxStream<'a, Result<MessageOutput, String>>;
}

impl_downcast!(LanguageModel);

#[derive(Clone)]
pub struct ArcMutexLanguageModel {
    pub model: Arc<Mutex<dyn LanguageModel>>,
    pub type_id: TypeId,
}

impl ArcMutexLanguageModel {
    pub fn new<T: LanguageModel + 'static>(model: T) -> Self {
        Self {
            model: Arc::new(Mutex::new(model)),
            type_id: TypeId::of::<T>(),
        }
    }

    pub fn new_from_arc<T: LanguageModel + 'static>(model: Arc<Mutex<T>>) -> Self {
        Self {
            model,
            type_id: TypeId::of::<T>(),
        }
    }

    pub fn into_inner<T: LanguageModel + 'static>(&self) -> Result<Arc<Mutex<T>>, String> {
        if TypeId::of::<T>() != self.type_id {
            return Err(format!(
                "TypeId of provided type is not same as {:?}",
                self.type_id
            ));
        }
        let arc_ptr = Arc::into_raw(self.model.clone());
        let arc_model = unsafe { Arc::from_raw(arc_ptr as *const Mutex<T>) };
        Ok(arc_model)
    }
}
