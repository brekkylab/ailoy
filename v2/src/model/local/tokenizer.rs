use std::{pin::Pin, str::FromStr};

use tokenizers::tokenizer::Tokenizer as HFTokenizer;

use crate::cache::{Cache, CacheElement, TryFromCache};

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
    ) -> Pin<Box<dyn Future<Output = Result<Vec<CacheElement>, String>>>> {
        let dirname = key.as_ref().replace("/", "--");
        Box::pin(async move { Ok(vec![CacheElement::new(dirname, "tokenizer.json")]) })
    }

    fn try_from_files(_: &Cache, files: Vec<(CacheElement, Vec<u8>)>) -> Result<Self, String> {
        let v = files.get(0).unwrap();
        let v = std::str::from_utf8(&v.1).map_err(|_| "Utf-8 conversion failed".to_owned())?;
        Ok(Tokenizer::new(v))
    }
}
