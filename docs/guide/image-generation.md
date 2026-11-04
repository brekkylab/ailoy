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

Only these three orgs are built in; a model of your own runs through [`ImageModel.custom`](#a-model-of-your-own).

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
use ailoy::{message::PartImage, model::create_image_model};

let model = create_image_model("google/gemini-3.1-flash-image").await?;
let output = model.generate("a red cube on a white table", None).await?;

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

The providers' image APIs share almost no option: OpenAI sizes an image in pixels, Gemini by an aspect ratio and a size class. So the options are one API's parameters, under that API's own names, written as one flat object.

| API | Read by | Fields |
| --- | --- | --- |
| OpenAI | `openai/…` | `n`, `size`, `quality`, `output_format`, `background`, `extra` |
| Gemini | `google/…` | `aspect_ratio`, `image_size`, `thinking_level`, `extra` |
| OpenRouter | `openrouter/…` | `n`, `size`, `resolution`, `aspect_ratio`, `quality`, `output_format`, `background`, `extra` |

A model reads the options as its own API's parameters, and a field its API does not have is an error naming the field, raised before anything is sent: `aspect_ratio` for an `openai/…` model, `n` for a `google/…` one. Options whose fields another API also has work there too, so `{"size": "1024x1024", "quality": "high"}` serves an `openai/…` and an `openrouter/…` model alike.

Values are written as each API's documentation gives them (`"1536x1024"`, `"16:9"`, `"2K"`, `"high"`) and sent as written. Which values a model takes is up to the model: a value it refuses comes back as the API's error. Where that error names the value only by its wire field (Gemini's size, ratio and thinking level, OpenRouter's `resolution`), a note in front of it names the option, such as `` `aspect_ratio: auto` is not accepted by 'gemini-3.1-flash-lite-image' ``.

::: code-group

```python [Python]
model = await ImageModel.create("openai/gpt-image-2")
output = await model.generate(
    "a red cube on a white table",
    options={"size": "1536x1024", "quality": "high", "output_format": "webp"},
)

model = await ImageModel.create("google/gemini-3.1-flash-image")
output = await model.generate(
    "a red cube on a white table",
    options={"aspect_ratio": "16:9", "image_size": "2K"},
)
```

```js [Node.js]
let model = await ImageModel.create('openai/gpt-image-2')
let output = await model.generate('a red cube on a white table', {
  size: '1536x1024',
  quality: 'high',
  output_format: 'webp',
})

model = await ImageModel.create('google/gemini-3.1-flash-image')
output = await model.generate('a red cube on a white table', {
  aspect_ratio: '16:9',
  image_size: '2K',
})
```

```rust [Rust]
use ailoy::model::{GeminiImageOptions, OpenAIImageOptions, create_image_model};

let model = create_image_model("openai/gpt-image-2").await?;
let options = OpenAIImageOptions {
    size: Some("1536x1024".into()),
    quality: Some("high".into()),
    output_format: Some("webp".into()),
    ..Default::default()
};
let output = model.generate("a red cube on a white table", Some(&options)).await?;

let model = create_image_model("google/gemini-3.1-flash-image").await?;
let options = GeminiImageOptions {
    aspect_ratio: Some("16:9".into()),
    image_size: Some("2K".into()),
    ..Default::default()
};
let output = model.generate("a red cube on a white table", Some(&options)).await?;
```

:::

For OpenAI and OpenRouter, two settings are refused before anything is sent, because no model takes them: `n` outside 1 to 10, and a `transparent` background with `jpeg` output, which cannot carry one.

### Other parameters: `extra`

For a parameter the fields above don't cover, put it in `extra`, written as the API's own JSON. It is merged into that API's request: objects key by key, anything else replacing what was there, so `extra` wins over a field that sets the same thing.

::: code-group

```python [Python]
# openai/…
options = {"output_format": "webp", "extra": {"output_compression": 80, "moderation": "low"}}
# google/…
options = {"extra": {"generationConfig": {"seed": 42}}}
# openrouter/…
options = {"extra": {"seed": 7, "provider": {"order": ["google-ai-studio"]}}}
```

```js [Node.js]
// openai/…
let options = { output_format: 'webp', extra: { output_compression: 80, moderation: 'low' } }
// google/…
options = { extra: { generationConfig: { seed: 42 } } }
// openrouter/…
options = { extra: { seed: 7, provider: { order: ['google-ai-studio'] } } }
```

```rust [Rust]
use ailoy::to_value;

let options = OpenAIImageOptions {
    output_format: Some("webp".into()),
    extra: Some(to_value!({"output_compression": 80, "moderation": "low"})),
    ..Default::default()
};
```

:::

The Gemini request nests its settings under `generationConfig`, so its `extra` does too. `extra` may not set what Ailoy sets itself: `model`, `prompt` and `stream` for OpenAI and OpenRouter, `contents` and `generationConfig.responseModalities` for Gemini.

## A model of your own

A model the three orgs do not cover, such as a local server, runs as an `ImageModel` too. Give `ImageModel.custom` an object with `generate(prompt, options)` that returns an image output, or a promise of one; in Rust, implement `InferImageModel`. The options it gets are what `generate` was given, with whatever keys your model reads; without any, Python and Node.js pass an empty object and Rust passes `None`. The output needs at least one image, or the call fails.

::: code-group

```python [Python]
class LocalModel:
    async def generate(self, prompt, options):
        steps = options.get("steps", 20)
        data = await render(prompt, steps)  # your own call
        return {"images": [{"type": "embedded", "mime_type": "image/png", "data": data}]}

model = ImageModel.custom(LocalModel())
output = await model.generate("a red cube on a white table", options={"steps": 30})
```

```js [Node.js]
const model = ImageModel.custom({
  async generate(prompt, options) {
    const steps = options.steps ?? 20
    const data = await render(prompt, steps) // your own call, giving a Buffer
    return { images: [{ type: 'embedded', mime_type: 'image/png', data }] }
  },
})
const output = await model.generate('a red cube on a white table', { steps: 30 })
```

```rust [Rust]
use ailoy::model::{ImageOptions, ImageOutput, InferImageModel};
use futures::future::BoxFuture;

#[derive(Default, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct LocalOptions {
    steps: Option<u32>,
}

struct LocalModel;

impl InferImageModel for LocalModel {
    fn generate(
        &self,
        prompt: &str,
        options: Option<&dyn ImageOptions>,
    ) -> BoxFuture<'static, anyhow::Result<ImageOutput>> {
        // A key `LocalOptions` does not have is an error naming it.
        let options = options.map(|o| o.read::<LocalOptions>()).transpose();
        let prompt = prompt.to_owned();
        Box::pin(async move {
            let options = options?.unwrap_or_default();
            render(&prompt, options.steps.unwrap_or(20)).await // your own call
        })
    }
}

let model = LocalModel;
let options = serde_json::json!({"steps": 30});
let output = model.generate("a red cube on a white table", Some(&options)).await?;
```

:::

## Retries

A rate limit (429) or an overloaded server (503, 529) is retried up to three times, waiting as the response's `retry-after` asks, else 1, 2 then 4 seconds, and never more than 10 seconds at a time. A 429 from OpenAI or Gemini that reports a spent quota is not retried. Other server errors are not either, since the image may already have been made and billed; OpenRouter alone also retries 502, which it documents as a failed generation it does not bill.
