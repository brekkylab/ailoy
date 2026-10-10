pub(crate) mod r#impl;
mod infer_lang_model;

pub use r#impl::{ClaudeModel, CodexModel, GeminiModel};
pub use infer_lang_model::*;

pub enum LangModel {
    Claude(ClaudeModel),
    Codex(CodexModel),
    Gemini(GeminiModel),
}
