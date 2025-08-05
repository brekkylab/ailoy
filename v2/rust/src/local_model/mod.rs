mod chat_template;
mod tokenizer;
mod tvm_model;

pub use chat_template::*;
pub use tokenizer::*;

#[derive(Debug)]
pub struct Inferencer {}

#[derive(Debug)]
pub struct LocalLangModel<'a> {
    pub chat_template: ChatTemplate<'a>,
    pub tokenizer: Tokenizer,
    pub inferencer: Inferencer,
}
