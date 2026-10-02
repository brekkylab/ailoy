# Laya

An agent answers typed decision questions (a choice, a score, a yes/no) about a text such as an email or a ticket, using the [Laya](https://huggingface.co/convaiinnovations/laya) decision model on the guest's GPU.

## Requirements

- GPU
- Python (with uv)

## Running

Rust (from the repository root):

```sh
cargo run --example laya
```

Python:

```sh
cd examples/laya/python
uv run main.py
```

Node:

```sh
cd examples/laya/node
npm install
npm start
```
