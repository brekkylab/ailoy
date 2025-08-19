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

######################
# ailoy (cxx bridge) #
######################
# @jhlee: Can we remove hard-coded paths?
set(CARGO_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(AILOY_CXXBRIDGE_RS ${CARGO_SOURCE_DIR}/src/ffi/cxx_bridge.rs)
set(AILOY_CXXBRIDGE_DIR ${CMAKE_CURRENT_BINARY_DIR}/cxx_bridge)
set(AILOY_CXXBRIDGE_CC ${AILOY_CXXBRIDGE_DIR}/cxx_bridge.rs.cc)
set(AILOY_CXXBRIDGE_H ${AILOY_CXXBRIDGE_DIR}/cxx_bridge.rs.h)
set(AILOY_CXXBRIDGE_BASE_DIR ${AILOY_CXXBRIDGE_DIR}/rust)
set(AILOY_CXXBRIDGE_BASE_H ${AILOY_CXXBRIDGE_BASE_DIR}/cxx.h)
file(MAKE_DIRECTORY ${AILOY_CXXBRIDGE_DIR})
file(MAKE_DIRECTORY ${AILOY_CXXBRIDGE_BASE_DIR})

# Create header file
execute_process(
    COMMAND cxxbridge ${AILOY_CXXBRIDGE_RS} --header
    OUTPUT_FILE ${AILOY_CXXBRIDGE_H}
    WORKING_DIRECTORY ${CARGO_SOURCE_DIR}
    RESULT_VARIABLE ret
)
if(NOT ret EQUAL 0)
    message(FATAL_ERROR "cxxbridge header generation failed ${AILOY_CXXBRIDGE_RS}")
endif()

# Create source file
execute_process(
    COMMAND cxxbridge ${AILOY_CXXBRIDGE_RS}
    OUTPUT_FILE ${AILOY_CXXBRIDGE_CC}
    WORKING_DIRECTORY ${CARGO_SOURCE_DIR}
    RESULT_VARIABLE ret
)
if(NOT ret EQUAL 0)
    message(FATAL_ERROR "cxxbridge cpp generation failed")
endif()

# Get base header dir
execute_process(
    COMMAND cxxbridge --header
    OUTPUT_FILE ${AILOY_CXXBRIDGE_BASE_H}
    WORKING_DIRECTORY ${CARGO_SOURCE_DIR}
    RESULT_VARIABLE ret
)
if(NOT ret EQUAL 0)
    message(FATAL_ERROR "cxxbridge cpp generation failed")
endif()

##################
# ailoy_cpp_shim #
##################
file(GLOB_RECURSE AILOY_CPP_SHIM_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp ${AILOY_CXXBRIDGE_CC})
add_library(ailoy_cpp_shim STATIC ${AILOY_CPP_SHIM_SRCS})
target_include_directories(ailoy_cpp_shim PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(ailoy_cpp_shim PUBLIC ${AILOY_CXXBRIDGE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/include)

#######
# tvm #
#######
if(APPLE)
    set(USE_METAL ON)
    set(USE_VULKAN OFF)
    target_compile_definitions(ailoy_cpp_shim PUBLIC USE_METAL=1)
elseif(WIN32)
    set(USE_METAL OFF)
    set(USE_VULKAN ON)
    target_compile_definitions(ailoy_cpp_shim PUBLIC USE_VULKAN=1)
elseif(LINUX)
    set(USE_METAL OFF)
    set(USE_VULKAN ON)
    target_compile_definitions(ailoy_cpp_shim PUBLIC USE_VULKAN=1)
else()
    message(FATAL_ERROR "Not implemented")
endif()

# Define TVM root if not already defined
include(ExternalProject)
ExternalProject_Add(tvm_build
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/tvm
    BUILD_COMMAND ${CMAKE_COMMAND} --build . --target tvm_runtime --parallel 8
    INSTALL_COMMAND ""
    CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DUSE_METAL=${USE_METAL}
        -DUSE_VULKAN=${USE_VULKAN}
        -DUSE_LIBBACKTRACE=OFF
        -DBUILD_DUMMY_LIBTVM=ON
)

# Include and link TVM headers/libraries manually
ExternalProject_Get_Property(tvm_build BINARY_DIR)
set(TVM_BINARY_DIR ${BINARY_DIR})

target_include_directories(ailoy_cpp_shim PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/tvm/include
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/tvm/ffi/include
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/tvm/3rdparty/picojson
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/tvm/3rdparty/dlpack/include
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/tvm/3rdparty/dmlc-core/include
)

add_library(tvm_runtime SHARED IMPORTED GLOBAL)
set_target_properties(tvm_runtime PROPERTIES
    IMPORTED_LOCATION ${TVM_BINARY_DIR}/libtvm_runtime.dylib
)
add_dependencies(tvm_runtime tvm_build)
target_link_libraries(ailoy_cpp_shim PUBLIC tvm_runtime)
target_compile_definitions(ailoy_cpp_shim PUBLIC
    DMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>
    AILOY_USE_TVM=1
)


if(AILOY_WITH_TEST)
    add_executable(test_tvm ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_tvm.cpp)
    target_link_libraries(test_tvm PRIVATE ailoy_cpp_shim)
    set_target_properties(test_tvm PROPERTIES OUTPUT_NAME "test_tvm")
    target_link_libraries(test_tvm PRIVATE GTest::gtest)
    target_link_options(test_tvm PRIVATE -fsanitize=undefined -fsanitize=address)
endif()

###########
# Install #
###########

install(TARGETS ailoy_cpp_shim ARCHIVE DESTINATION .)
install(FILES ${TVM_BINARY_DIR}/libtvm_runtime.dylib DESTINATION .)
