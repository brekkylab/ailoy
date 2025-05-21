cmake_minimum_required(VERSION 3.15)
project(ailoy_py C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(FetchContent)

FetchContent_Declare(pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.13.6
)
FetchContent_MakeAvailable(pybind11)

find_package(ailoy REQUIRED)

file(GLOB_RECURSE AILOY_PY_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/csrc/*.cpp)
pybind11_add_module(ailoy_py ${AILOY_PY_SRCS})
target_link_libraries(ailoy_py PRIVATE ailoy::ailoy)
set_target_properties(ailoy_py PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${SETUPTOOLS_EXTDIR}"
)

if(APPLE)
    # If using Homebrew LLVM, explicitly tell the compiler to use the macOS system SDK.
    execute_process(
        COMMAND xcrun --sdk macosx --show-sdk-path
        OUTPUT_VARIABLE MACOS_SDK_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    target_compile_options(ailoy_py PUBLIC -isysroot ${MACOS_SDK_PATH})
    target_link_options(ailoy_py PUBLIC -isysroot ${MACOS_SDK_PATH})
endif()
