include(FetchContent)

file(GLOB AILOY_VM_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
if(AILOY_WITH_TVM)
    file(GLOB_RECURSE AILOY_VM_TVM_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/tvm/*.cpp)
    list(APPEND AILOY_VM_SRCS ${AILOY_VM_TVM_SRCS})
endif()

add_library(ailoy_vm_obj OBJECT ${AILOY_VM_SRCS})
target_include_directories(ailoy_vm_obj PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(ailoy_vm_obj PUBLIC ailoy_broker_client_obj ailoy_core_obj)

# indicators
FetchContent_Declare(
    indicators
    URL https://github.com/p-ranav/indicators/archive/ac6c93ea2b1f97a220d10a0729a625b3f51e320b.tar.gz
    EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(indicators)
target_link_libraries(ailoy_vm_obj PRIVATE indicators::indicators)

# minijinja
set(MINIJINJA_BUILD_SHARED OFF)
set(MINIJINJA_INSTALL OFF)
set(MINIJINJA_RUST_FLAGS "${RUSTFLAGS}")
set(MINIJINJA_CARGO_BUILD_FLAGS "${CARGO_BUILD_FLAGS}")
FetchContent_Declare(
    minijinja
    GIT_REPOSITORY https://github.com/brekkylab/minijinja.git
    GIT_TAG 2671e3b0d8817f499177e669f4edf8d5b870950a
    SOURCE_SUBDIR minijinja-cabi
)
FetchContent_MakeAvailable(minijinja)
target_link_libraries(ailoy_vm_obj PUBLIC minijinja::cabi)

# tinyexpr
set(TINYEXPR_ROOT ${CMAKE_BINARY_DIR}/_deps/tinyexpr)
add_compile_definitions(TE_NAT_LOG)
file(DOWNLOAD https://raw.githubusercontent.com/codeplea/tinyexpr/9907207e5def0fabdb60c443517b0d9e9d521393/tinyexpr.h ${TINYEXPR_ROOT}/include/tinyexpr.h)
file(DOWNLOAD https://raw.githubusercontent.com/codeplea/tinyexpr/9907207e5def0fabdb60c443517b0d9e9d521393/tinyexpr.c ${TINYEXPR_ROOT}/src/tinyexpr.c)
add_library(tinyexpr STATIC ${TINYEXPR_ROOT}/src/tinyexpr.c)
# Note that tinyexpr is compiled with the C compiler (not CXX).
# While this is generally not an issue, it can cause problems during LTO on some compilers.
set_target_properties(tinyexpr PROPERTIES INTERPROCEDURAL_OPTIMIZATION FALSE)
target_include_directories(tinyexpr PUBLIC ${TINYEXPR_ROOT}/include)
target_link_libraries(ailoy_vm_obj PUBLIC tinyexpr)

# magic_enum
target_link_libraries(ailoy_vm_obj PRIVATE magic_enum)

# tokenizers-cpp
set(TOKENIZERS_CPP_RUST_FLAGS "${RUSTFLAGS}")
set(TOKENIZERS_CPP_CARGO_FLAGS "${CARGO_BUILD_FLAGS}")
if(NODE)
    set(TOKENIZERS_CPP_MSVC_RUNTIME_LIBRARY "MT" CACHE STRING "" FORCE)
endif()
FetchContent_Declare(
    tokenizers-cpp
    # GIT_REPOSITORY https://github.com/mlc-ai/tokenizers-cpp.git
    GIT_REPOSITORY https://github.com/brekkylab/tokenizers-cpp.git
    GIT_TAG a8934c69c01338de39fa4f05c4ef25b596fc552f
    EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(tokenizers-cpp)
target_link_libraries(ailoy_vm_obj PUBLIC tokenizers_c tokenizers_cpp)
if(EMSCRIPTEN)
    # Need to disable TOKENIZERS_PARALLELISM on WASM
    # https://github.com/mlc-ai/tokenizers-cpp/pull/42
    target_compile_definitions(tokenizers_cpp PRIVATE COMPILE_WASM_RUNTIME)
endif()


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
            target_compile_definitions(ailoy_vm_obj PUBLIC USE_METAL=1)
        elseif(WIN32)
            set(USE_VULKAN ON)
            target_compile_definitions(ailoy_vm_obj PUBLIC USE_VULKAN=1)
        elseif(LINUX)
            set(USE_VULKAN ON)
            target_compile_definitions(ailoy_vm_obj PUBLIC USE_VULKAN=1)
        else()
            message(FATAL_ERROR "Not implemented")
        endif()
        set(BUILD_DUMMY_LIBTVM ON)
        set(USE_LIBBACTRACE OFF)
        FetchContent_MakeAvailable(tvm)
        set(TVM_SOURCE_DIR ${tvm_SOURCE_DIR})
        set(TVM_BINARY_DIR ${tvm_BINARY_DIR})
    endif()
    target_include_directories(ailoy_vm_obj PUBLIC
        ${TVM_SOURCE_DIR}/3rdparty/dmlc-core/include
        ${TVM_SOURCE_DIR}/ffi/include
        ${TVM_SOURCE_DIR}/include
    )
    target_link_libraries(tvm_runtime PRIVATE dlpack)
    target_link_directories(ailoy_vm_obj PUBLIC ${TVM_BINARY_DIR})
    target_link_libraries(ailoy_vm_obj PRIVATE tvm_runtime)
    target_compile_definitions(ailoy_vm_obj PUBLIC DMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>)
    target_compile_definitions(ailoy_vm_obj PUBLIC AILOY_USE_TVM=1)

    # xgrammar
    FetchContent_Declare(
        xgrammar
        URL https://github.com/mlc-ai/xgrammar/archive/refs/tags/v0.1.19.tar.gz
        EXCLUDE_FROM_ALL
    )
    # Since xgrammar overrides settings via a config file, calling `set` in here has no effect.
    # Instead, create a `config.cmake` file to set the configuration.
    if(NOT EXISTS "${CMAKE_BINARY_DIR}/config.cmake")
    file(WRITE "${CMAKE_BINARY_DIR}/config.cmake" "")
    endif()
    file(APPEND "${CMAKE_BINARY_DIR}/config.cmake" "set(XGRAMMAR_BUILD_PYTHON_BINDINGS OFF)\n")
    FetchContent_MakeAvailable(xgrammar)
    target_link_libraries(xgrammar PRIVATE dlpack)
    target_link_libraries(ailoy_vm_obj PRIVATE xgrammar)
endif()

# faiss
if(DEFINED EMSCRIPTEN)
    FetchContent_Declare(
        faiss
        URL https://github.com/brekkylab/faiss-web/releases/download/v1.11.0-wasm/faiss-v1.11.0-wasm.tar.gz
        URL_HASH SHA256=b10884dfc61025d3de3f1c26565e5a0601d54b247e6b4abcc8408472d93479d6
        DOWNLOAD_EXTRACT_TIMESTAMP YES
        EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(faiss)
    set(faiss_DIR "${faiss_SOURCE_DIR}/share/faiss")
    find_package(faiss REQUIRED CONFIG)
else()
    find_package(OpenMP)
    if(NOT OpenMP_FOUND)
        message(FATAL_ERROR "OpenMP not found. You have to specify OpenMP Root directory by either -DOpenMP_ROOT option or `OpenMP_ROOT` environment variable")
    endif()
    FetchContent_Declare(
        faiss
        GIT_REPOSITORY https://github.com/facebookresearch/faiss.git
        GIT_TAG v1.11.0
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
endif()
target_link_libraries(ailoy_vm_obj PUBLIC faiss)

if(AILOY_WITH_TEST)
    add_executable(test_debug_module ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_debug_module.cpp)
    target_include_directories(test_debug_module PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    add_test(NAME TestDebugModule COMMAND test_debug_module)
    target_link_libraries(test_debug_module PRIVATE ailoy_vm_obj ailoy_broker_client_obj ailoy_core_obj GTest::gtest)
    target_link_options(test_debug_module PRIVATE -fsanitize=undefined -fsanitize=address)

    add_executable(test_vm ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_vm.cpp)
    target_include_directories(test_vm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    add_test(NAME TestVM COMMAND test_vm)
    target_link_libraries(test_vm PRIVATE ailoy_vm_obj ailoy_broker_obj ailoy_broker_client_obj ailoy_core_obj GTest::gtest)
    target_link_options(test_vm PRIVATE -fsanitize=undefined -fsanitize=address)
    
    if(AILOY_WITH_TVM)
        add_executable(test_tvm_language_model ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_tvm_language_model.cpp)
        add_test(NAME TestTVMLanguageModel COMMAND test_tvm_language_model)
        target_include_directories(test_tvm_language_model PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
        target_link_libraries(test_tvm_language_model PRIVATE ailoy_vm_obj ailoy_broker_obj ailoy_broker_client_obj ailoy_core_obj GTest::gtest GTest::gmock)
        target_link_options(test_tvm_language_model PRIVATE -fsanitize=undefined -fsanitize=address)

        add_executable(test_tvm_embedding_model ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_tvm_embedding_model.cpp)
        add_test(NAME TestTVMEmbeddingModel COMMAND test_tvm_embedding_model)
        target_include_directories(test_tvm_embedding_model PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
        target_link_libraries(test_tvm_embedding_model PRIVATE ailoy_vm_obj ailoy_broker_obj ailoy_broker_client_obj ailoy_core_obj GTest::gtest)
        target_link_options(test_tvm_embedding_model PRIVATE -fsanitize=undefined)
    endif()

    add_executable(test_model_cache ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_model_cache.cpp)
    add_test(NAME TestModelCache COMMAND test_model_cache)
    target_include_directories(test_model_cache PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(test_model_cache PRIVATE ailoy_vm_obj ailoy_core_obj)
    if(EMSCRIPTEN)
        configure_file("${CMAKE_CURRENT_SOURCE_DIR}/tests/test_model_cache.html" "${CMAKE_CURRENT_BINARY_DIR}/test_model_cache.html" COPYONLY)
    else()
        # NOTE: Do not add sanitizer options on Emscripten. It leads to poor performance!
        target_link_options(test_model_cache PRIVATE -fsanitize=undefined -fsanitize=address)
        target_link_libraries(test_model_cache PRIVATE GTest::gtest)
    endif()

    add_executable(test_faiss_vector_store ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_faiss_vector_store.cpp)
    add_test(NAME TestFaissVectorStore COMMAND test_faiss_vector_store)
    target_include_directories(test_faiss_vector_store PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(test_faiss_vector_store PRIVATE ailoy_vm_obj ailoy_broker_obj ailoy_broker_client_obj ailoy_core_obj GTest::gtest)
    if(EMSCRIPTEN)
        configure_file("${CMAKE_CURRENT_SOURCE_DIR}/tests/test_faiss_vector_store.html" "${CMAKE_CURRENT_BINARY_DIR}/test_faiss_vector_store.html" COPYONLY)
    else()
        # NOTE: Do not add sanitizer options on Emscripten. It leads to poor performance!
        target_link_options(test_faiss_vector_store PRIVATE -fsanitize=undefined -fsanitize=address)
        target_link_libraries(test_faiss_vector_store PRIVATE GTest::gtest)
    endif()

    add_executable(test_chromadb_vector_store ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_chromadb_vector_store.cpp)
    add_test(NAME TestChromadbVectorStore COMMAND test_chromadb_vector_store)
    target_include_directories(test_chromadb_vector_store PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(test_chromadb_vector_store PRIVATE ailoy_vm_obj ailoy_broker_obj ailoy_broker_client_obj ailoy_core_obj GTest::gtest)
    target_link_options(test_chromadb_vector_store PRIVATE -fsanitize=undefined -fsanitize=address)

    add_executable(test_split_text ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_split_text.cpp)
    add_test(NAME TestSplitText COMMAND test_split_text)
    target_link_libraries(test_split_text PRIVATE ailoy_vm_obj ailoy_broker_obj ailoy_broker_client_obj ailoy_core_obj GTest::gtest)
    target_link_options(test_split_text PRIVATE -fsanitize=undefined -fsanitize=address)

    add_executable(test_calculator ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_calculator.cpp)
    add_test(NAME TestCalculator COMMAND test_calculator)
    target_link_libraries(test_calculator PRIVATE ailoy_vm_obj ailoy_broker_obj ailoy_broker_client_obj ailoy_core_obj tinyexpr GTest::gtest)
    target_link_options(test_calculator PRIVATE -fsanitize=undefined -fsanitize=address)

    add_executable(test_http_request ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_http_request.cpp)
    add_test(NAME TestHttpRequest COMMAND test_http_request)
    target_include_directories(test_http_request PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(test_http_request PRIVATE ailoy_vm_obj ailoy_broker_obj ailoy_broker_client_obj ailoy_core_obj nlohmann_json::nlohmann_json GTest::gtest)
    target_link_options(test_http_request PRIVATE -fsanitize=undefined -fsanitize=address)

    add_executable(test_api_models ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_api_models.cpp)
    add_test(NAME TestAPIModels COMMAND test_api_models)
    target_include_directories(test_api_models PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(test_api_models PRIVATE ailoy_vm_obj ailoy_broker_obj ailoy_broker_client_obj ailoy_core_obj nlohmann_json::nlohmann_json GTest::gtest GTest::gmock)
    target_link_options(test_api_models PRIVATE -fsanitize=undefined -fsanitize=address)

    add_executable(test_tokenizers ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_tokenizers.cpp)
    target_include_directories(test_tokenizers PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(test_tokenizers PRIVATE ailoy_vm_obj ailoy_core_obj)
    if(EMSCRIPTEN)
        configure_file("${CMAKE_CURRENT_SOURCE_DIR}/tests/test_tokenizers.html" "${CMAKE_CURRENT_BINARY_DIR}/test_tokenizers.html" COPYONLY)
    else()
        # NOTE: Do not add sanitizer options on Emscripten. It leads to poor performance!
        target_link_options(test_tokenizers PRIVATE -fsanitize=undefined -fsanitize=address)
        target_link_libraries(test_tokenizers PRIVATE GTest::gtest)
    endif()
endif()
