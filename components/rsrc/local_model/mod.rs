mod chat_template;
mod tokenizer;
mod tvm_model;

pub use chat_template::*;
pub use tokenizer::*;
pub use tvm_model::*;

use crate::{LangModel, Message};

#[derive(Debug)]
pub struct Inferencer {}

#[derive(Debug)]
pub struct LocalLangModel {
    pub chat_template: ChatTemplate,
    pub tokenizer: Tokenizer,
    pub inferencer: Inferencer,
}

impl LangModel for LocalLangModel {
    fn init(&mut self, model_name: &str) -> () {}

    fn infer<T: Iterator<Item = String>>(&self, _: &Vec<Message>) -> T {
        todo!()
    }
}
