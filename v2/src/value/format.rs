pub trait Format: Clone {}

pub struct FormattedRef<'inner, D, F: Format> {
    pub inner: &'inner D,
    _marker: std::marker::PhantomData<F>,
}

impl<'inner, D, F: Format> FormattedRef<'inner, D, F> {
    pub fn new(inner: &'inner D) -> Self {
        Self {
            inner,
            _marker: std::marker::PhantomData,
        }
    }
}

pub struct FormattedOwned<D, F: Format> {
    pub inner: D,
    _marker: std::marker::PhantomData<F>,
}

impl<D, F: Format> FormattedOwned<D, F> {
    pub fn new(inner: D) -> Self {
        Self {
            inner,
            _marker: std::marker::PhantomData,
        }
    }
}

// mod qwen {
//     use crate::value::{Format, MessageFormat, PartFormat, ToolCallFormat};

//     #[derive(Clone)]
//     pub struct QwenToolCallFormat;

//     impl Format for QwenToolCallFormat {}

//     impl ToolCallFormat for QwenToolCallFormat {}

//     #[derive(Clone)]
//     pub struct QwenPartFormat;

//     impl Format for QwenPartFormat {}

//     impl PartFormat for QwenPartFormat {
//         type ToolCall = QwenToolCallFormat;
//     }

//     #[derive(Clone)]
//     pub struct QwenMessageFormat;

//     impl Format for QwenMessageFormat {}

//     impl MessageFormat for QwenMessageFormat {
//         type Part = QwenPartFormat;
//         const REASONING_FIELD: &'static str = "reasoning_content";
//         const CONTENTS_TEXTONLY: &'static bool = &true;
//     }
// }

// pub use qwen::*;

// mod openai {
//     use crate::value::{Format, MessageFormat, PartFormat, ToolCallFormat};

//     #[derive(Clone)]
//     pub struct OpenAIToolCallFormat;

//     impl Format for OpenAIToolCallFormat {}

//     impl ToolCallFormat for OpenAIToolCallFormat {
//         const FUNCTION_ARGUMENTS_FIELD: &'static str = "parameters";
//     }

//     #[derive(Clone)]
//     pub struct OpenAIPartFormat;

//     impl Format for OpenAIPartFormat {}

//     impl PartFormat for OpenAIPartFormat {
//         type ToolCall = OpenAIToolCallFormat;
//         const IMAGE_URL_FIELD: &'static str = "image_url";
//         const AUDIO_URL_FIELD: &'static str = "audio_url";
//     }

//     #[derive(Clone)]
//     pub struct OpenAIMessageFormat;

//     impl Format for OpenAIMessageFormat {}

//     impl MessageFormat for OpenAIMessageFormat {
//         type Part = OpenAIPartFormat;
//     }
// }

// pub use openai::*;
