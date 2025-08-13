cmake_minimum_required(VERSION 3.28)

# Compatibility with CMake < 3.5 has been removed from CMake>=4.0.
# Add -DCMAKE_POLICY_VERSION_MINIMUM=3.5 to try configuring anyway.
if(CMAKE_VERSION VERSION_GREATER_EQUAL "4.0.0")
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
endif()

# Avoid FetchContent warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24:
cmake_policy(SET CMP0135 NEW)

##################
# Global options #
##################
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(APPLE)
    # Support OSX>=14.0
    set(CMAKE_OSX_DEPLOYMENT_TARGET 14.0)

    # If using Homebrew LLVM, explicitly tell the compiler to use the macOS system SDK.
    execute_process(
        COMMAND xcrun --sdk macosx --show-sdk-path
        OUTPUT_VARIABLE MACOS_SDK_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(CMAKE_OSX_SYSROOT "${MACOS_SDK_PATH}")
endif()

# Some platform-specific global options
if(APPLE)
    add_compile_definitions(APPLE)
elseif(UNIX)
    add_compile_definitions(UNIX)
elseif(MSVC)
    add_compile_options(/utf-8)
endif()

#################
# Project ailoy #
#################
project(ailoy C CXX)

set(AILOY_WITH_TEST ON CACHE BOOL "Build test")
if(DEFINED EMSCRIPTEN)
    set(AILOY_WITH_TVM OFF CACHE BOOL "Enable tvm support")

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions -pthread")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -s ASYNCIFY=1 -s FETCH=1 -s ALLOW_MEMORY_GROWTH=1 -s WASM_BIGINT=1 -fexceptions -lembind")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -s EXPORTED_RUNTIME_METHODS=['ccall','cwrap','stringToNewUTF8']")
    set(RUST_NIGHTLY ON)
    set(RUSTFLAGS "-Ctarget-feature=+atomics,+bulk-memory")
    set(CARGO_BUILD_FLAGS "-Z;build-std=std,panic_abort")
else()
    set(RUST_NIGHTLY OFF)
    set(AILOY_WITH_TVM ON CACHE BOOL "Enable tvm support")
endif()
set(TVM_ROOT "" CACHE PATH "Root directory of mlc-llm. If AILOY_WITH_TVM is ON and this is empty, it will be downloaded automatically.")

include(FetchContent)

FetchContent_Declare(magic_enum URL https://github.com/Neargye/magic_enum/archive/refs/tags/v0.9.7.tar.gz EXCLUDE_FROM_ALL)
FetchContent_MakeAvailable(magic_enum)

if(AILOY_WITH_TEST)
    FetchContent_Declare(googletest URL https://github.com/google/googletest/archive/refs/tags/v1.16.0.tar.gz)
    set(INSTALL_GTEST OFF)
    if(WIN32)
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    endif()
    FetchContent_MakeAvailable(googletest)
endif()

##################
# Subdirectories #
##################
add_subdirectory(core)
add_subdirectory(components)
add_subdirectory(broker)
add_subdirectory(vm)

add_subdirectory(bindings/python)
add_subdirectory(bindings/js-node)
add_subdirectory(bindings/js-web)
