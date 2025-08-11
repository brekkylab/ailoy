use std::{pin::Pin, str::FromStr};

use tokenizers::tokenizer::Tokenizer as HFTokenizer;

use crate::cache::{Cache, CacheContents, CacheEntry, TryFromCache};

#[derive(Debug, Clone)]
pub struct Tokenizer {
    inner: HFTokenizer,
}

impl Tokenizer {
    pub fn new(config: &str) -> Self {
        Tokenizer {
            inner: HFTokenizer::from_str(config).unwrap().into(),
        }
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, String> {
        let encoded = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(|e| format!("Tokenizer::encode failed: {}", e.to_string()))?;
        Ok(encoded.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, String> {
        self.inner
            .decode(ids, skip_special_tokens)
            .map_err(|e| format!("Tokenizer::decode failed: {}", e.to_string()))
    }
}

impl TryFromCache for Tokenizer {
    fn claim_files(
        _: Cache,
        key: impl AsRef<str>,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<CacheEntry>, String>>>> {
        let dirname = key.as_ref().replace("/", "--");
        Box::pin(async move { Ok(vec![CacheEntry::new(dirname, "tokenizer.json")]) })
    }

    fn try_from_contents(contents: &mut CacheContents) -> Result<Self, String> {
        let Some((_, bytes)) = contents.remove_with_filename("tokenizer.json") else {
            return Err("tokenizer.json not exists".to_owned());
        };
        let s = std::str::from_utf8(&bytes).map_err(|_| "Utf-8 conversion failed".to_owned())?;
        Ok(Tokenizer::new(s))
    }
}
