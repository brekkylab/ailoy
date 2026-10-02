# CAD

An agent designs 3D parts in [CadQuery](https://cadquery.readthedocs.io). It writes the script, renders the model from four sides, looks at the pictures and fixes the model until it matches the request. The output is STEP, STL and GLB files.

## Running

Rust (from the repository root):

```sh
cargo run --example cad
```

Python:

```sh
cd examples/cad/python
uv run main.py
```

Node:

```sh
cd examples/cad/node
npm install
npm start
```
