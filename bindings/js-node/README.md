# ailoy-js-node

Node.js binding for Ailoy runtime APIs

## Install

> [!WARNING]
> We are currently undergoing review for package managers (PyPI / npm) for upload.
> In the meantime, if you'd like to try it out, please use the wheel files provided.
> [https://github.com/brekkylab/ailoy/releases/tag/v0.0.1](https://github.com/brekkylab/ailoy/releases/tag/v0.0.1)

```bash
# npm
npm install ailoy-js-node
# yarn
yarn add ailoy-js-node
```

## Building from source

### Prerequisites

- Node 18 or higher
- Python 3.10 or higher
- C/C++ compiler
  (recommended versions are below)
  - GCC >= 13
  - LLVM Clang >= 17
  - Apple Clang >= 15
  - MSVC >= 19.29
- CMake >= 3.24.0
- Git
- OpenSSL
- Rust & Cargo >= 1.82.0
- OpenMP
- BLAS
- LAPACK
- Vulkan SDK (if you are using vulkan)

```bash
cd bindings/js-node
npm run build
```
