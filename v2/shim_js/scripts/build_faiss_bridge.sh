#!/bin/bash

# Prepare directory to download and extract Faiss web
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BUILD_DIR="$SCRIPT_DIR/../build"
FAISS_DEPS_DIR="$BUILD_DIR/deps/faiss"
mkdir -p "$FAISS_DEPS_DIR" || { echo "Error: Could not create directory $FAISS_DEPS_DIR"; exit 1; }

# # Download and extract
FAISS_WEB_URL="https://github.com/brekkylab/faiss-web/releases/download/v1.11.0-wasm/faiss-v1.11.0-wasm.tar.gz"
curl -L -s "$FAISS_WEB_URL" | tar -xzf - -C "$FAISS_DEPS_DIR"

# Build faiss_bridge
docker run --rm \
    -v "$SCRIPT_DIR/..":/workspace \
    -v $BUILD_DIR/.emscripten_cache:/opt/emsdk/upstream/emscripten/cache \
    -e EMCC_SKIP_SANITY_CHECK=1 \
    --entrypoint /opt/emsdk/upstream/emscripten/emcc \
    ghcr.io/r-wasm/flang-wasm:v20.1.4 \
    /workspace/csrc/faiss_bridge.cpp \
    -I/workspace/build/deps/faiss/include \
    -L/workspace/build/deps/faiss/lib \
    -L/opt/flang/wasm/lib \
    -lfaiss \
    -lblas \
    -llapack \
    -lFortranRuntime \
    -lembind \
    -s WASM=1 \
    -s EXPORT_ES6=1 \
    -s MODULARIZE=1 \
    -s EXPORT_NAME="FaissModule" \
    -s ALLOW_MEMORY_GROWTH=1 \
    -s DISABLE_EXCEPTION_CATCHING=0 \
    --emit-tsd faiss_bridge.d.ts \
    -o /workspace/src/faiss/faiss_bridge.js
