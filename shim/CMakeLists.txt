cmake_minimum_required(VERSION 3.28)

# Compatibility with CMake < 3.5 has been removed from CMake>=4.0.
# Add -DCMAKE_POLICY_VERSION_MINIMUM=3.5 to try configuring anyway.
if(CMAKE_VERSION VERSION_GREATER_EQUAL "4.0.0")
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
endif()

# Avoid FetchContent warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24:
cmake_policy(SET CMP0135 NEW)

include(FetchContent)

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

set(AILOY_WITH_TEST OFF CACHE BOOL "Build test")

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
set(AILOY_CXXBRIDGE_RS ${PROJECT_SOURCE_DIR}/../src/ffi/cxx_bridge.rs CACHE FILEPATH "Path to the cxx_bridge.rs")
set(AILOY_CXXBRIDGE_OUTDIR ${PROJECT_BINARY_DIR}/cxxbridge CACHE PATH "Output dir for cxxbridge generated files")
set(AILOY_CXXBRIDGE_CC ${AILOY_CXXBRIDGE_OUTDIR}/cxx_bridge.rs.cc)
set(AILOY_CXXBRIDGE_H ${AILOY_CXXBRIDGE_OUTDIR}/cxx_bridge.rs.h)
set(AILOY_CXXBRIDGE_BASE_H ${AILOY_CXXBRIDGE_OUTDIR}/rust/cxx.h)

# Ensure the cxxbridge CLI is available (cargo install cxxbridge-cmd)
find_program(CXXBRIDGE_EXE NAMES cxxbridge REQUIRED)

# Create output directories at configure time
file(MAKE_DIRECTORY ${AILOY_CXXBRIDGE_OUTDIR})
file(MAKE_DIRECTORY ${AILOY_CXXBRIDGE_OUTDIR}/rust)

# ---- Helper script to capture stdout -> file (portable) ----
# cxxbridge prints to stdout; CMake COMMAND has no portable redirection.
# We call a small CMake script that uses execute_process(OUTPUT_FILE ...).
set(_CXXBRIDGE_SCRIPT ${CMAKE_CURRENT_BINARY_DIR}/_cxxbridge.cmake)
file(WRITE "${_CXXBRIDGE_SCRIPT}" [=[
  if(NOT DEFINED MODE)
    message(FATAL_ERROR "MODE not set (header|cc|base_header)")
  endif()
  if(NOT DEFINED CXXBRIDGE_EXE)
    message(FATAL_ERROR "cxxbridge-cmd not cound. Please install with `cargo install cxxbridge-cmd`.")
  endif()
  if("${MODE}" STREQUAL "header")
    execute_process(COMMAND ${CXXBRIDGE_EXE} ${INPUT} --header OUTPUT_FILE ${OUTPUT} RESULT_VARIABLE rv COMMAND_ECHO STDOUT)
  elseif("${MODE}" STREQUAL "cc")
    execute_process(COMMAND ${CXXBRIDGE_EXE} ${INPUT} OUTPUT_FILE ${OUTPUT} RESULT_VARIABLE rv COMMAND_ECHO STDOUT)
  elseif("${MODE}" STREQUAL "base_header")
    execute_process(COMMAND ${CXXBRIDGE_EXE} --header OUTPUT_FILE ${OUTPUT} RESULT_VARIABLE rv COMMAND_ECHO STDOUT)
  else()
    message(FATAL_ERROR "Unknown MODE='${MODE}'")
  endif()
  if(NOT rv EQUAL 0)
    message(FATAL_ERROR "cxxbridge failed: ${rv}")
  endif()
]=])

# ---- Per-artifact custom commands (rebuilds only when inputs change) ----
# Generate the C++ header from the Rust bridge
add_custom_command(
    OUTPUT ${AILOY_CXXBRIDGE_H}
    COMMAND ${CMAKE_COMMAND}
        -DCXXBRIDGE_EXE=${CXXBRIDGE_EXE}
        -DINPUT=${AILOY_CXXBRIDGE_RS}
        -DOUTPUT=${AILOY_CXXBRIDGE_H}
        -DMODE=header
        -P ${_CXXBRIDGE_SCRIPT}
    DEPENDS ${AILOY_CXXBRIDGE_RS}
    COMMENT "cxxbridge: generating ${AILOY_CXXBRIDGE_H}"
    VERBATIM
)

# Generate the C++ source from the Rust bridge
add_custom_command(
    OUTPUT "${AILOY_CXXBRIDGE_CC}"
    COMMAND ${CMAKE_COMMAND}
        -DCXXBRIDGE_EXE=${CXXBRIDGE_EXE}
        -DINPUT=${AILOY_CXXBRIDGE_RS}
        -DOUTPUT=${AILOY_CXXBRIDGE_CC}
        -DMODE=cc
        -P ${_CXXBRIDGE_SCRIPT}
    DEPENDS ${AILOY_CXXBRIDGE_RS}
    COMMENT "cxxbridge: generating ${AILOY_CXXBRIDGE_CC}"
    VERBATIM
)

# Generate the base support header (cxx.h)
add_custom_command(
    OUTPUT ${AILOY_CXXBRIDGE_BASE_H}
    COMMAND ${CMAKE_COMMAND}
        -DCXXBRIDGE_EXE=${CXXBRIDGE_EXE}
        -DOUTPUT=${AILOY_CXXBRIDGE_BASE_H}
        -DMODE=base_header
        -P ${_CXXBRIDGE_SCRIPT}
    COMMENT "cxxbridge: generating ${AILOY_CXXBRIDGE_BASE_H}"
    VERBATIM
)

# Aggregate target so others can depend on "generation done"
add_custom_target(ailoy_cxxbridge_gen
  DEPENDS "${AILOY_CXXBRIDGE_H}" "${AILOY_CXXBRIDGE_CC}" "${AILOY_CXXBRIDGE_BASE_H}"
)

# Mark generated for IDEs
set_source_files_properties(${AILOY_CXXBRIDGE_H} PROPERTIES GENERATED TRUE)
set_source_files_properties(${AILOY_CXXBRIDGE_CC} PROPERTIES GENERATED TRUE)
set_source_files_properties(${AILOY_CXXBRIDGE_BASE_H} PROPERTIES GENERATED TRUE)

##################
# ailoy_cpp_shim #
##################
file(GLOB_RECURSE AILOY_CPP_SHIM_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
add_library(ailoy_cpp_shim STATIC ${AILOY_CPP_SHIM_SRCS})
add_dependencies(ailoy_cpp_shim ailoy_cxxbridge_gen)
target_sources(ailoy_cpp_shim PRIVATE ${AILOY_CXXBRIDGE_CC})
target_include_directories(ailoy_cpp_shim PUBLIC ${AILOY_CXXBRIDGE_OUTDIR} ${CMAKE_CURRENT_SOURCE_DIR}/include)

#######
# tvm #
#######
FetchContent_Declare(
    tvm
    GIT_REPOSITORY https://github.com/brekkylab/relax.git
    GIT_TAG f27bdf6b8371484ec0444a2a83dfaa2fe7c27fe3
    EXCLUDE_FROM_ALL
)
set(CONDA_BUILD ON) # Actually it is not related to conda, but this is only way to disable python build
set(BUILD_DUMMY_LIBTVM ON)
set(BUILD_STATIC_RUNTIME ON)
set(USE_LIBBACTRACE OFF)
set(TVM_FFI_USE_LIBBACKTRACE OFF)
if(APPLE)
    set(USE_METAL ON)
    target_compile_definitions(ailoy_cpp_shim PUBLIC USE_METAL=1)
elseif(WIN32)
    set(USE_VULKAN ON)
    target_compile_definitions(ailoy_cpp_shim PUBLIC USE_VULKAN=1)
elseif(LINUX)
    set(USE_VULKAN ON)
    target_compile_definitions(ailoy_cpp_shim PUBLIC USE_VULKAN=1)
else()
    message(FATAL_ERROR "Not implemented")
endif()
FetchContent_MakeAvailable(tvm)

# Hollow wrapper for TVM runtime.
# The static TVM runtime must be linked with --whole-archive at the final link step.
# Since `ailoy_cpp_shim` is an intermediate static library, we must NOT pull TVM symbols
# into it. This INTERFACE target mirrors TVM’s usage requirements (include dirs,
# compile definitions/options) without adding `tvm_runtime` to the link line.
# Consumers (final executables/shared libs) should link `tvm_runtime` themselves, e.g.:
#   target_link_libraries(my_exe PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,tvm_runtime>")
add_library(tvm_runtime_hollow INTERFACE)
target_include_directories(tvm_runtime_hollow INTERFACE
    $<TARGET_PROPERTY:tvm_runtime,INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:tvm_runtime,INTERFACE_INCLUDE_DIRECTORIES>
)
target_compile_definitions(tvm_runtime_hollow INTERFACE
    $<TARGET_PROPERTY:tvm_runtime,COMPILE_DEFINITIONS>
    $<TARGET_PROPERTY:tvm_runtime,INTERFACE_COMPILE_DEFINITIONS>
)
target_compile_options(tvm_runtime_hollow INTERFACE
    $<TARGET_PROPERTY:tvm_runtime,COMPILE_OPTIONS>
    $<TARGET_PROPERTY:tvm_runtime,INTERFACE_COMPILE_OPTIONS>
)
add_dependencies(tvm_runtime_hollow tvm_runtime tvm_ffi_static)

# User of ailoy_cpp_shim should import tvm_runtime with whole-archive options
target_link_libraries(ailoy_cpp_shim PRIVATE tvm_runtime_hollow)

if(AILOY_WITH_TEST)
    add_executable(test_tvm ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_tvm.cpp)
    target_link_libraries(test_tvm PRIVATE ailoy_cpp_shim)
    set_target_properties(test_tvm PROPERTIES OUTPUT_NAME "test_tvm")
    target_link_libraries(test_tvm PRIVATE GTest::gtest)
    target_link_options(test_tvm PRIVATE -fsanitize=undefined -fsanitize=address)
endif()

#########
# faiss #
#########
find_package(OpenMP)
if(NOT OpenMP_FOUND)
    message(FATAL_ERROR "OpenMP not found. You have to specify OpenMP Root directory by either -DOpenMP_ROOT option or `OpenMP_ROOT` environment variable")
endif()
FetchContent_Declare(
    faiss
    GIT_REPOSITORY https://github.com/facebookresearch/faiss.git
    GIT_TAG v1.12.0
    EXCLUDE_FROM_ALL
)
find_package(MKL QUIET)
if (MKL_FOUND)
    message(STATUS "Enable MKL Support")
    set(BLA_STATIC ON CACHE BOOL "Use static libraries for BLAS" FORCE)
    set(FAISS_ENABLE_MKL ON CACHE BOOL "Enable MKL support" FORCE)
else()
    message(STATUS "Disable MKL Support")
    set(FAISS_ENABLE_MKL OFF CACHE BOOL "Enable MKL support" FORCE)
endif()
set(FAISS_ENABLE_GPU OFF CACHE BOOL "Disable GPU support" FORCE)
set(FAISS_ENABLE_PYTHON OFF CACHE BOOL "Disable Python bindings" FORCE)
set(FAISS_ENABLE_EXTRAS OFF CACHE BOOL "Disable extra components" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "Disable build tests" FORCE)
FetchContent_MakeAvailable(faiss)
target_link_libraries(ailoy_cpp_shim PUBLIC faiss)

###########
# Install #
###########

install(TARGETS ailoy_cpp_shim ARCHIVE DESTINATION .)
install(TARGETS tvm_runtime ARCHIVE DESTINATION .)
install(TARGETS tvm_ffi_static ARCHIVE DESTINATION .)
install(TARGETS faiss ARCHIVE DESTINATION .)
