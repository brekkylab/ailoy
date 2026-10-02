# SAM3

An agent segments images and videos with [SAM3](https://huggingface.co/facebook/sam3), run through ncnn on the guest's GPU (Vulkan). It can find objects from a text prompt, segment one object from points or a box, and track objects through a video.

## Requirements

- GPU
- Python (with uv)

## Running

Rust (from the repository root):

```sh
cargo run --example sam3
```

Python:

```sh
cd examples/sam3/python
uv run main.py
```

Node:

```sh
cd examples/sam3/node
npm install
npm start
```
