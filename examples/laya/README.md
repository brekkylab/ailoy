# Laya

An agent routes a support inbox to departments (engineering, finance, sales, legal, marketing), asking the [Laya](https://huggingface.co/convaiinnovations/laya) decision model on the guest's GPU where each ticket belongs. Each ticket in `shared/tickets/` is filed under `artifacts/routes/<department>/`, with a summary in `artifacts/routing.md`.

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
