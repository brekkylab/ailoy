# Ailoy Python Runtime

## Install
```bash
pip install ailoy
```

## Building from source

### Prerequisites

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


### Setup development environment

```bash
pip install -e .
```

### Generate wheel

```bash
python -m build -w
```
