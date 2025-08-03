use crate::Message;

pub trait LangModel {
    fn init(&mut self, model_name: &str) -> ();

    fn infer<T: Iterator<Item = String>>(&self, messages: &Vec<Message>) -> T;
}

pub struct APILangModel {
    url: String,
    api_key: String,
}

impl LangModel for APILangModel {
    fn init(&mut self, model_name: &str) -> () {}

    fn infer<T: Iterator<Item = String>>(&self, messages: &Vec<Message>) -> T {
        todo!()
    }
}
