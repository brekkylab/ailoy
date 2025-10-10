use std::str::FromStr;

use tokenizers::tokenizer::Tokenizer as HFTokenizer;

use crate::{
    cache::{Cache, CacheClaim, CacheContents, TryFromCache},
    utils::BoxFuture,
};
use anyhow::{Context, anyhow, bail};

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

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> anyhow::Result<Vec<u32>> {
        let encoded = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(|e| anyhow!("Tokenizer::encode failed: {}", e))?;
        Ok(encoded.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> anyhow::Result<String> {
        self.inner
            .decode(ids, skip_special_tokens)
            .map_err(|e| anyhow!("Tokenizer::decode failed: {}", e))
    }
}

impl TryFromCache for Tokenizer {
    fn claim_files(
        _: Cache,
        key: impl AsRef<str>,
    ) -> BoxFuture<'static, anyhow::Result<CacheClaim>> {
        let dirname = key.as_ref().replace("/", "--");
        Box::pin(async move { Ok(CacheClaim::new([(dirname.as_str(), "tokenizer.json")])) })
    }

    fn try_from_contents(mut contents: CacheContents) -> BoxFuture<'static, anyhow::Result<Self>> {
        Box::pin(async move {
            let Some((_, bytes)) = contents.remove_with_filename("tokenizer.json") else {
                bail!("tokenizer.json not exists");
            };
            let s = std::str::from_utf8(&bytes).context("Utf-8 conversion failed")?;
            Ok(Tokenizer::new(s))
        })
    }
}
