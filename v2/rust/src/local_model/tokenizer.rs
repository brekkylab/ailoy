use std::{path::PathBuf, pin::Pin, str::FromStr};

use tokenizers::tokenizer::Tokenizer as HFTokenizer;

use crate::cache::{Cache, TryFromCache};

#[derive(Debug)]
pub struct Tokenizer {
    inner: HFTokenizer,
}

impl Tokenizer {
    pub fn new(config: &str) -> Self {
        Tokenizer {
            inner: HFTokenizer::from_str(config).unwrap().into(),
        }
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Vec<u32> {
        let encoded = self.inner.encode(text, add_special_tokens).unwrap();
        return encoded.get_ids().to_vec();
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> String {
        self.inner.decode(ids, skip_special_tokens).unwrap()
    }
}

impl TryFromCache for Tokenizer {
    fn claim_files(
        _: Cache,
        key: impl AsRef<str>,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<PathBuf>, String>>>> {
        let dir = key.as_ref().replace("/", "--");
        Box::pin(async move { Ok(vec![PathBuf::from(dir).join("chat_template.j2".to_owned())]) })
    }

    fn try_from_files(files: Vec<(PathBuf, Vec<u8>)>) -> Result<Self, String> {
        let v = files.get(0).unwrap();
        let v = std::str::from_utf8(&v.1).map_err(|_| "Utf-8 conversion failed".to_owned())?;
        Ok(Tokenizer::new(v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const QWEN3_TOKENIZER: &str = include_str!("../../../data/Qwen--Qwen3-0.6B/tokenizer.json");

    #[test]
    fn test_qwen3_encode() {
        let tokenizer = Tokenizer::new(QWEN3_TOKENIZER);
        let prompt: &str = r#"<|im_start|>system
You are an assistant.<|im_end|>
<|im_start|>user
Hi what's your name?<|im_end|>
<|im_start|>assistant
<think>

</think>

You can call me Jaden.<|im_end|>
<|im_start|>user
Who made you?<|im_end|>
<|im_start|>assistant
<think>

</think>

"#;
        let expected: Vec<u32> = vec![
            151644, 8948, 198, 2610, 525, 458, 17847, 13, 151645, 198, 151644, 872, 198, 13048,
            1128, 594, 697, 829, 30, 151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271,
            2610, 646, 1618, 752, 619, 21140, 13, 151645, 198, 151644, 872, 198, 15191, 1865, 498,
            30, 151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271,
        ];
        let result = tokenizer.encode(prompt, true);
        assert_eq!(expected, result);
    }
}
