# Retail Bench

Runs [RetailBench](https://github.com/linghuazhang01/RetailBench) against an agent. The agent manages a supermarket one day per turn, ordering stock and setting prices, for as long as the store survives.

## Requirements

- Python (with uv)

## Running

Rust (from the repository root):

```sh
cargo run --example retail_bench
```

Python:

```sh
cd examples/retail_bench/python
uv run main.py
```

Node:

```sh
cd examples/retail_bench/node
npm install
npm start
```
