# Image Generation

An image model turns a text prompt into one or more images. It is built from a name the way a language model is, and runs on its own: an agent does not call it.

## Naming a model

Name the model as `<org>/<name>`:

| Org | Runs through | Name | Example | Needs |
| --- | --- | --- | --- | --- |
| `openai` | OpenAI's Images API | The official API id | `openai/gpt-image-2` | `OPENAI_API_KEY` |
| `google` | The Gemini API | The official API id | `google/gemini-3.1-flash-image` | `GEMINI_API_KEY` |
| `openrouter` | OpenRouter | OpenRouter's model slug, as is | `openrouter/bytedance-seed/seedream-5-0-pro` | `OPENROUTER_API_KEY` |

Image models run only through their org's own API; the [backends](./model-providers#backends) of language models do not apply. To reach a model through OpenRouter, name it under `openrouter/`, which works next to the org's own API: `google/gemini-3.1-flash-image` and `openrouter/google/gemini-3.1-flash-image` are the same model, reached two ways.

## Generating an image

`ImageModel` builds the model, and `generate` runs it on a prompt. The output holds at least one image; a call that ends with none, such as one stopped by a safety filter, raises with the reason the provider gave.

::: code-group

```python [Python]
from ailoy import ImageModel

model = await ImageModel.create("google/gemini-3.1-flash-image")
output = await model.generate("a red cube on a white table")

image = output["images"][0]
with open("cube.jpg", "wb") as f:
    f.write(image["data"])
```

```js [Node.js]
const fs = require('node:fs')
const { ImageModel } = require('@brekkylab/ailoy')

const model = await ImageModel.create('google/gemini-3.1-flash-image')
const output = await model.generate('a red cube on a white table')

const image = output.images[0]
fs.writeFileSync('cube.jpg', image.data)
```

```rust [Rust]
use ailoy::{
    message::PartImage,
    model::{ImageModelOptions, create_image_model},
};

let model = create_image_model("google/gemini-3.1-flash-image").await?;
let output = model.generate("a red cube on a white table", &ImageModelOptions::default()).await?;

if let PartImage::Embedded { data, .. } = &output.images[0] {
    std::fs::write("cube.jpg", data)?;
}
```

:::

The output has

- `images`: normally embedded images, each with its `mime_type` and its `data` (`bytes` in Python, a `Buffer` in Node.js), in the encoding the provider chose or you asked for. A provider that answers with a link gives a `url` instead, which may not last;
- `text`: any text the model returned with the images;
- `usage`: the tokens the call was billed for, when the provider reports them.

## Options

The providers' image APIs share almost no option: OpenAI sizes an image in pixels, Gemini by an aspect ratio and a size class. So the options come in one **section** per API, each holding that API's parameters under its own names, and a model reads only its own section. Fill several sections to switch models without changing the options.

| Section | Read by | Fields |
| --- | --- | --- |
| `openai` | `openai/…` | `n`, `size`, `quality`, `output_format`, `background`, `extra` |
| `gemini` | `google/…` | `aspect_ratio`, `image_size`, `thinking_level`, `extra` |
| `openrouter` | `openrouter/…` | `n`, `size`, `resolution`, `aspect_ratio`, `quality`, `output_format`, `background`, `extra` |

Values are written as each API's documentation gives them (`"1536x1024"`, `"16:9"`, `"2K"`, `"high"`) and sent as written. Which values a model takes is up to the model: a value it refuses comes back as the API's error, led by a line naming the option, such as `` `gemini.aspect_ratio: auto` is not accepted by 'gemini-3.1-flash-lite-image' ``. Unknown sections and keys are ignored.

::: code-group

```python [Python]
options = {
    "openai": {"size": "1536x1024", "quality": "high", "output_format": "webp"},
    "gemini": {"aspect_ratio": "16:9", "image_size": "2K"},
    "openrouter": {"aspect_ratio": "16:9", "resolution": "2K"},
}

for name in ["openai/gpt-image-2", "google/gemini-3.1-flash-image"]:
    model = await ImageModel.create(name)
    output = await model.generate("a red cube on a white table", options=options)
```

```js [Node.js]
const options = {
  openai: { size: '1536x1024', quality: 'high', output_format: 'webp' },
  gemini: { aspect_ratio: '16:9', image_size: '2K' },
  openrouter: { aspect_ratio: '16:9', resolution: '2K' },
}

for (const name of ['openai/gpt-image-2', 'google/gemini-3.1-flash-image']) {
  const model = await ImageModel.create(name)
  const output = await model.generate('a red cube on a white table', options)
}
```

```rust [Rust]
use ailoy::model::{
    GeminiImageOptions, ImageModelOptions, OpenAiImageOptions, OpenRouterImageOptions,
    create_image_model,
};

let options = ImageModelOptions {
    openai: Some(OpenAiImageOptions {
        size: Some("1536x1024".into()),
        quality: Some("high".into()),
        output_format: Some("webp".into()),
        ..Default::default()
    }),
    gemini: Some(GeminiImageOptions {
        aspect_ratio: Some("16:9".into()),
        image_size: Some("2K".into()),
        ..Default::default()
    }),
    openrouter: Some(OpenRouterImageOptions {
        aspect_ratio: Some("16:9".into()),
        resolution: Some("2K".into()),
        ..Default::default()
    }),
};

for name in ["openai/gpt-image-2", "google/gemini-3.1-flash-image"] {
    let model = create_image_model(name).await?;
    let output = model.generate("a red cube on a white table", &options).await?;
}
```

:::

In the `openai` and `openrouter` sections, two settings are refused before anything is sent, because no model takes them: `n` outside 1 to 10, and a `transparent` background with `jpeg` output, which cannot carry one.

### Other parameters: `extra`

For a parameter the fields above don't cover, put it in the section's `extra`, written as the API's own JSON. It is merged into that API's request: objects key by key, anything else replacing what was there, so `extra` wins over a field that sets the same thing.

::: code-group

```python [Python]
options = {
    "openai": {"output_format": "webp", "extra": {"output_compression": 80, "moderation": "low"}},
    "gemini": {"extra": {"generationConfig": {"seed": 42}}},
    "openrouter": {"extra": {"seed": 7, "provider": {"order": ["google-ai-studio"]}}},
}
```

```js [Node.js]
const options = {
  openai: { output_format: 'webp', extra: { output_compression: 80, moderation: 'low' } },
  gemini: { extra: { generationConfig: { seed: 42 } } },
  openrouter: { extra: { seed: 7, provider: { order: ['google-ai-studio'] } } },
}
```

```rust [Rust]
use ailoy::to_value;

let options = ImageModelOptions {
    openai: Some(OpenAiImageOptions {
        output_format: Some("webp".into()),
        extra: Some(to_value!({"output_compression": 80, "moderation": "low"})),
        ..Default::default()
    }),
    gemini: Some(GeminiImageOptions {
        extra: Some(to_value!({"generationConfig": {"seed": 42}})),
        ..Default::default()
    }),
    ..Default::default()
};
```

:::

The Gemini request nests its settings under `generationConfig`, so its `extra` does too. `extra` may not set what Ailoy sets itself: `model`, `prompt` and `stream` for OpenAI and OpenRouter, `contents` and `generationConfig.responseModalities` for Gemini.

## Retries

A rate limit (429) or an overloaded server (503, 529) is retried up to three times, waiting as the response's `retry-after` asks, else 1, 2 then 4 seconds, and never more than 10 seconds at a time. A spent quota is not retried. Other server errors are not either, since the image may already have been made and billed; OpenRouter alone also retries 502, which it documents as a failed generation it does not bill.

## Bringing your own model

As with [language models](./model-providers#bringing-your-own-model), an org can be served by your own code: give `set_image_model_provider` (`setImageModelProvider`) a factory. It is called with the org and the name after it, and returns a model with `generate(prompt, options)`, which returns the output (or, in Python and Node.js, an awaitable or promise of it). The factory may also return an `ImageModel`.

::: code-group

```python [Python]
import ailoy
from ailoy import ImageModel

PNG_BYTES = open("blank.png", "rb").read()

class Blank:
    async def generate(self, prompt, options):
        return {"images": [{"type": "embedded", "mime_type": "image/png", "data": PNG_BYTES}]}

ailoy.set_image_model_provider("mine", lambda org, name: Blank())
model = await ImageModel.create("mine/blank")
```

```js [Node.js]
const fs = require('node:fs')
const { ImageModel, setImageModelProvider } = require('@brekkylab/ailoy')

const PNG_BYTES = fs.readFileSync('blank.png')

class Blank {
  async generate(prompt, options) {
    return { images: [{ type: 'embedded', mime_type: 'image/png', data: PNG_BYTES }] }
  }
}

setImageModelProvider('mine', (org, name) => new Blank())
const model = await ImageModel.create('mine/blank')
```

```rust [Rust]
use std::sync::Arc;

use ailoy::{
    message::PartImage,
    model::{
        ImageModelOptions, ImageModelProvider, ImageOutput, InferImageModel, create_image_model,
        set_image_model_provider,
    },
};
use futures::future::BoxFuture;

const PNG_BYTES: &[u8] = include_bytes!("blank.png");

struct Blank;

impl InferImageModel for Blank {
    fn generate(
        &self,
        _prompt: &str,
        _options: &ImageModelOptions,
    ) -> BoxFuture<'static, anyhow::Result<ImageOutput>> {
        Box::pin(async {
            Ok(ImageOutput {
                images: vec![PartImage::Embedded {
                    mime_type: "image/png".into(),
                    data: PNG_BYTES.to_vec().into(),
                }],
                text: None,
                usage: None,
            })
        })
    }
}

set_image_model_provider("mine", ImageModelProvider::custom(|_org, _name| Ok(Arc::new(Blank))));
let model = create_image_model("mine/blank").await?;
```

:::

In Python and Node.js the factory itself must return the model directly, not an awaitable or promise of it.

`get_image_model_provider` (`getImageModelProvider`) reads an org's provider back, and `remove_image_model_provider` (`removeImageModelProvider`) takes it away. The image table is separate from the language one: setting one does not change the other.
