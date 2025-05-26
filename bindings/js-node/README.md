# ailoy-node

Node.js binding for Ailoy runtime APIs

## Install
```bash
# npm
npm install ailoy-node
# yarn
yarn add ailoy-node
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
