mod chat_template;
mod inferencer;
mod tokenizer;

use std::{pin::Pin, sync::Arc};

use async_stream::try_stream;
use futures::Stream;
use tokio::sync::{Mutex, RwLock};

use crate::{Message, MessageDelta, Part, language_model::LanguageModel};

pub use chat_template::*;
pub use inferencer::*;
pub use tokenizer::*;

#[derive(Debug, Clone)]
pub struct LocalLanguageModel {
    chat_template: Arc<RwLock<ChatTemplate>>,
    tokenizer: Arc<RwLock<Tokenizer>>,
    inferencer: Arc<Mutex<Inferencer>>,
}

impl LanguageModel for LocalLanguageModel {
    fn run(self, msgs: Vec<Message>) -> Pin<Box<dyn Stream<Item = Result<MessageDelta, String>>>> {
        let strm = try_stream! {
            let prompt = self.chat_template.read().await.apply_with_vec(&msgs, true)?;
            let input_tokens = self.tokenizer.read().await.encode(&prompt, true);
            self.inferencer.lock().await.prefill(&input_tokens);
            let mut last_token = *input_tokens.last().unwrap();
            // @jhlee: TODO complete decode logic
            for _ in 0..10 {
                let new_token = self.inferencer.lock().await.decode(last_token);
                let v = vec![new_token];
                let s = self.tokenizer.read().await.decode(v.as_slice(), false);
                yield MessageDelta::content(Part::Text(s));
                last_token = new_token;
            }
        };
        Box::pin(strm)
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
