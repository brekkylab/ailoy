# Offshore Leaks

An agent analyses ICIJ's [Offshore Leaks database](https://offshoreleaks.icij.org) (Panama, Paradise and Pandora Papers, and more). The data is loaded into DuckDB, and the agent writes and runs its own SQL and Python to answer questions, producing reports, tables and charts.

## Requirements

- Python (with uv)

## Running

Rust (from the repository root):

```sh
cargo run --example offshore_leaks
```

Python:

```sh
cd examples/offshore_leaks/python
uv run main.py
```

Node:

```sh
cd examples/offshore_leaks/node
npm install
npm start
```
