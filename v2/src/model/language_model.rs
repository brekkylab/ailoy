use downcast_rs::{Downcast, impl_downcast};
use dyn_clone::{DynClone, clone_trait_object};

use crate::{
    utils::{BoxStream, MaybeSend, MaybeSync},
    value::{Message, MessageOutput, ToolDesc},
};

pub trait LanguageModel: DynClone + Downcast + MaybeSend + MaybeSync {
    /// Runs the language model with the given tools and messages, returning a stream of `MessageOutput`s.
    fn run<'a>(
        self: &'a mut Self,
        msg: Vec<Message>,
        tools: Vec<ToolDesc>,
    ) -> BoxStream<'a, Result<MessageOutput, String>>;
}

clone_trait_object!(LanguageModel);
impl_downcast!(LanguageModel);
