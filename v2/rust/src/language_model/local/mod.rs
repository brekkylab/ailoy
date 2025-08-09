mod chat_template;
mod inferencer;
mod tokenizer;

pub use chat_template::*;
pub use inferencer::*;
pub use tokenizer::*;

use crate::language_model::LanguageModel;

#[derive(Debug)]
pub struct LocalLanguageModel<'a> {
    chat_template: ChatTemplate<'a>,
    tokenizer: Tokenizer,
    inferencer: Inferencer,
}

impl<'a> LanguageModel for LocalLanguageModel<'a> {
    fn run(
        &self,
        _msg: &Vec<crate::Message>,
    ) -> std::pin::Pin<Box<dyn futures::Stream<Item = Result<crate::MessageDelta, String>>>> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(any(target_family = "unix", target_family = "windows"))]
    #[tokio::test]
    async fn test1() {
        use crate::{Message, Part, Role};

        let cache = crate::cache::Cache::new();
        let key = "Qwen/Qwen3-0.6B";
        let ct = cache.try_create::<ChatTemplate>(key).await.unwrap();
        let tok = cache.try_create::<Tokenizer>(key).await.unwrap();
        let mut model = cache.try_create::<Inferencer>(key).await.unwrap();
        let msgs = vec![
            Message::with_content(Role::System, Part::from_text("You are an assistant.")),
            Message::with_content(Role::User, Part::from_text("Hi what's your name?")),
        ];
        let prompt = ct.apply_with_vec(&msgs, true).unwrap();
        println!("{:?}", prompt);
        let toks = tok.encode(&prompt, true);
        println!("{:?}", toks);
        let recovered = tok.decode(toks.as_slice(), false);
        println!("{:?}", recovered);
        model.prefill(&toks);
        let sampled = model.decode(*toks.last().unwrap());
        println!("{:?}", sampled);
        let sampled = model.decode(sampled);
        println!("{:?}", sampled);
        let sampled = model.decode(sampled);
        println!("{:?}", sampled);
    }
}
