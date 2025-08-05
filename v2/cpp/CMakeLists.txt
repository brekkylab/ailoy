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
project(ailoy CXX)

set(AILOY_WITH_TEST ON CACHE BOOL "Build test")
set(AILOY_WITH_TVM ON CACHE BOOL "Enable tvm support")
set(TVM_ROOT "" CACHE PATH "Root directory of tvm. If AILOY_WITH_TVM is ON and this is empty, it will be downloaded automatically.")

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
    enable_testing()
endif()

#############
# ailoy_cpp #
#############
file(GLOB_RECURSE AILOY_CORE_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
add_library(ailoy_cpp STATIC ${AILOY_CORE_SRCS})
target_include_directories(ailoy_cpp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# nlohmann_json
FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
    EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(nlohmann_json)
target_link_libraries(ailoy_cpp PRIVATE nlohmann_json)

# tvm
if(AILOY_WITH_TVM)
    if(TVM_ROOT)
        if(NOT DEFINED TVM_SOURCE_DIR)
            set(TVM_SOURCE_DIR ${TVM_ROOT})
        endif()
        if(NOT DEFINED TVM_BINARY_DIR)
            set(TVM_BINARY_DIR ${TVM_ROOT}/build)
        endif()
    else()
        FetchContent_Declare(
            tvm
            GIT_REPOSITORY https://github.com/brekkylab/relax.git
            GIT_TAG e2aa6d0ebf9bd82632a510c00f3966eeee89e68a
            EXCLUDE_FROM_ALL
        )
        if(APPLE)
            set(USE_METAL ON)
            target_compile_definitions(ailoy_cpp PUBLIC USE_METAL=1)
        elseif(WIN32)
            set(USE_VULKAN ON)
            target_compile_definitions(ailoy_cpp PUBLIC USE_VULKAN=1)
        elseif(LINUX)
            set(USE_VULKAN ON)
            target_compile_definitions(ailoy_cpp PUBLIC USE_VULKAN=1)
        else()
            message(FATAL_ERROR "Not implemented")
        endif()
        set(BUILD_DUMMY_LIBTVM ON)
        set(USE_LIBBACTRACE OFF)
        FetchContent_MakeAvailable(tvm)
        set(TVM_SOURCE_DIR ${tvm_SOURCE_DIR})
        set(TVM_BINARY_DIR ${tvm_BINARY_DIR})
    endif()
    target_include_directories(ailoy_cpp PUBLIC
        ${TVM_SOURCE_DIR}/3rdparty/dlpack/include
        ${TVM_SOURCE_DIR}/3rdparty/dmlc-core/include
        ${TVM_SOURCE_DIR}/ffi/include
        ${TVM_SOURCE_DIR}/include
    )
    target_link_directories(ailoy_cpp PUBLIC ${TVM_BINARY_DIR})
    target_link_libraries(ailoy_cpp PUBLIC tvm_runtime)
    target_compile_definitions(ailoy_cpp PUBLIC DMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>)
    target_compile_definitions(ailoy_cpp PUBLIC AILOY_USE_TVM=1)
endif()


if(AILOY_WITH_TEST)
    add_executable(test_tvm ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_tvm.cpp)
    target_link_libraries(test_tvm PRIVATE ailoy_cpp)
    set_target_properties(test_tvm PROPERTIES OUTPUT_NAME "test_tvm")
    target_link_libraries(test_tvm PRIVATE GTest::gtest)
    target_link_options(test_tvm PRIVATE -fsanitize=undefined -fsanitize=address)
endif()
