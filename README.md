# Ailoy

## Build from source

### Prerequsites

- C/C++ compiler
- CMake
- Git
- OpenSSL (libssl-dev)
- Rust & Cargo (optional, required for mlc-llm)
- OpenMP (libomp-dev) (optional, used by Faiss)
- BLAS (libblas-dev) (optional, used by Faiss)
- LAPACK (liblapack-dev) (optional, used by Faiss)

### Node.js Build

```bash
cd bindings/js-node
npm run build
```

See `bindings/js-node/README.md` for more details.

### Python Build

```bash
cd bindings/python
pip wheel --no-deps -w dist .
```

See `bindings/python/README.md` for more details.
