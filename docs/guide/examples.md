# Examples

Rust examples are in [`examples/`](https://github.com/brekkylab/ailoy/tree/main/examples) and run with `cargo run --example <name>`.
Python examples are in [`bindings/python/examples`](https://github.com/brekkylab/ailoy/tree/main/bindings/python/examples), and Node examples are in [`bindings/node/examples`](https://github.com/brekkylab/ailoy/tree/main/bindings/node/examples).

::: tip
You'll need a GPU (any GPU that supports Vulkan, or Metal on Macs) for the examples that run ML models.
:::

| Example | Description | Requirements |
| --- | --- | :---: |
| [hello](https://github.com/brekkylab/ailoy/tree/main/examples/hello) | One turn with no tools and no console | |
| [cad](https://github.com/brekkylab/ailoy/tree/main/examples/cad) | Writes CadQuery, renders the model from four sides, looks at the renders and iterates | |
| [offshore_leaks](https://github.com/brekkylab/ailoy/tree/main/examples/offshore_leaks) | Analyses the ICIJ Offshore Leaks database with SQL and Python that the agent writes itself | |
| [retail_bench](https://github.com/brekkylab/ailoy/tree/main/examples/retail_bench) | Runs a supermarket simulator, one day per turn | |
| [sam3](https://github.com/brekkylab/ailoy/tree/main/examples/sam3) | Segments images and videos with SAM3 on the guest GPU (ncnn + Vulkan) | GPU |
| [tts](https://github.com/brekkylab/ailoy/tree/main/examples/tts) | Speaks text in a voice described in words, using Qwen3-TTS | GPU |
| [laya](https://github.com/brekkylab/ailoy/tree/main/examples/laya) | Answers typed decision questions with a local model on the GPU | GPU |
