use std::{any::TypeId, sync::Arc};

use downcast_rs::{Downcast, impl_downcast};
use futures::lock::Mutex;

use crate::{
    utils::{BoxStream, MaybeSend, MaybeSync},
    value::{Config, Message, MessageDelta, ToolDesc},
};

pub trait LanguageModel: Downcast + MaybeSend + MaybeSync {
    /// Runs the language model with the given tools and messages, returning a stream of `MessageOutput`s.
    fn run<'a>(
        self: &'a mut Self,
        msg: Vec<Message>,
        tools: Vec<ToolDesc>,
        config: Config,
    ) -> BoxStream<'a, Result<MessageDelta, String>>;
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
