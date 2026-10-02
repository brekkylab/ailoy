# TTS(Text-to-Speech)

An agent speaks a text in a voice described in words, such as "a calm woman in her thirties, speaking slowly", using [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) on the guest's GPU. It reads the text and the voice description from the context folder and hands back a WAV file.

## Requirements

- GPU
- Python (with uv)

## Running

Rust (from the repository root):

```sh
cargo run --example tts
```

Python:

```sh
cd examples/tts/python
uv run main.py
```

Node:

```sh
cd examples/tts/node
npm install
npm start
```
