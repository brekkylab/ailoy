# Blender

An agent builds 3D scenes in a live [Blender](https://www.blender.org) while you watch over VNC. It drives Blender with Python, renders what it built, looks at the pictures and fixes the scene until it matches the request. The output is a `.blend`, a `.glb`, renders and a turntable video.

Open `vnc://localhost:5901` (Screen Sharing on macOS, or any VNC viewer) with the password `ailoy` once the run says Blender is up. Blender stays open after the agent finishes, until you press Enter.

Blender draws with Vulkan. It uses the host's GPU when that GPU has the features Blender needs, and otherwise renders on the guest's CPUs with Mesa's llvmpipe, which is slower. On a Mac it is always llvmpipe.

## Running

Rust (from the repository root):

```sh
cargo run --example blender
```

Python:

```sh
cd examples/blender/python
uv run main.py
```

Node:

```sh
cd examples/blender/node
npm install
npm start
```
