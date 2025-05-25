include(FetchContent)

file(GLOB_RECURSE AILOY_VM_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
if(AILOY_WITH_MLC_LLM)
    file(GLOB_RECURSE AILOY_VM_MLC_LLM_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/mlc_llm/*.cpp)
    list(APPEND AILOY_VM_SRCS ${AILOY_VM_MLC_LLM_SRCS})
endif()
if(AILOY_WITH_FAISS)
    file(GLOB_RECURSE AILOY_VM_FAISS_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/faiss/*.cpp)
    list(APPEND AILOY_VM_SRCS ${AILOY_VM_FAISS_SRCS})
endif()
add_library(ailoy_vm_obj OBJECT ${AILOY_VM_SRCS})
target_include_directories(ailoy_vm_obj PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(ailoy_vm_obj PUBLIC ailoy_broker_client_obj ailoy_core_obj)

# cpp-httplib
FetchContent_Declare(
    httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
    GIT_TAG v0.20.0
    EXCLUDE_FROM_ALL
)
set(HTTPLIB_REQUIRE_OPENSSL ON)
set(BROTLI_USE_STATIC_LIBS ON)
set(OPENSSL_USE_STATIC_LIBS ON)
if(NODE)
    set(OPENSSL_MSVC_STATIC_RT ON)
endif()
FetchContent_MakeAvailable(httplib)
target_link_libraries(ailoy_vm_obj PRIVATE httplib)

# openssl
find_package(OpenSSL REQUIRED)
target_link_libraries(ailoy_vm_obj PRIVATE OpenSSL::Crypto OpenSSL::SSL)

# indicators
FetchContent_Declare(
    indicators
    URL https://github.com/p-ranav/indicators/archive/ac6c93ea2b1f97a220d10a0729a625b3f51e320b.tar.gz
    EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(indicators)
target_link_libraries(ailoy_vm_obj PRIVATE indicators::indicators)

# minja
set(MINJA_ROOT ${CMAKE_BINARY_DIR}/_deps/minja)
file(DOWNLOAD https://raw.githubusercontent.com/google/minja/f06140fa52fd140fe38e531ec373d8dc9c86aa06/include/minja/minja.hpp ${MINJA_ROOT}/include/minja/minja.hpp)
file(DOWNLOAD https://raw.githubusercontent.com/google/minja/f06140fa52fd140fe38e531ec373d8dc9c86aa06/include/minja/chat-template.hpp ${MINJA_ROOT}/include/minja/chat-template.hpp)

target_include_directories(ailoy_vm_obj PRIVATE ${MINJA_ROOT}/include)

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

# mlc-llm
if(AILOY_WITH_MLC_LLM)
    if(MLC_LLM_ROOT)
        if(NOT DEFINED MLC_LLM_SOURCE_DIR)
            set(MLC_LLM_SOURCE_DIR ${MLC_LLM_ROOT})
        endif()
        if(NOT DEFINED MLC_LLM_BINARY_DIR)
            set(MLC_LLM_BINARY_DIR ${MLC_LLM_ROOT}/build)
        endif()
    else()
        FetchContent_Declare(
            mlc-llm
            GIT_REPOSITORY https://github.com/brekkylab/mlc-llm.git
            GIT_TAG 24b0b72296d64c3a5e4ce5f2efaabfda51e1a8b8
            EXCLUDE_FROM_ALL
        )
        if(NODE)
            set(TOKENIZERS_CPP_MSVC_RUNTIME_LIBRARY "MT" CACHE STRING "" FORCE)
        endif()

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
        FetchContent_MakeAvailable(mlc-llm)
        set(MLC_LLM_SOURCE_DIR ${mlc-llm_SOURCE_DIR})
        set(MLC_LLM_BINARY_DIR ${mlc-llm_BINARY_DIR})
    endif()
    if(MSVC)
        set_target_properties(mlc_llm PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
        set_target_properties(mlc_llm_static PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
    endif()
    target_include_directories(ailoy_vm_obj PUBLIC
        ${MLC_LLM_SOURCE_DIR}/3rdparty/tvm/3rdparty/dmlc-core/include
        ${MLC_LLM_SOURCE_DIR}/3rdparty/tvm/3rdparty/dlpack/include
        ${MLC_LLM_SOURCE_DIR}/3rdparty/tvm/3rdparty/picojson
        ${MLC_LLM_SOURCE_DIR}/3rdparty/tvm/include
        ${MLC_LLM_SOURCE_DIR}/3rdparty/tokenizers-cpp/include
        ${MLC_LLM_SOURCE_DIR}/3rdparty/xgrammar/include
        ${MLC_LLM_SOURCE_DIR}/cpp
    )
    target_link_directories(ailoy_vm_obj PUBLIC
        ${MLC_LLM_BINARY_DIR}
        ${MLC_LLM_BINARY_DIR}/tvm
    )
    target_link_libraries(ailoy_vm_obj PUBLIC
        tvm_runtime
        mlc_llm_static
        tokenizers_c
        tokenizers_cpp
    )
    target_compile_definitions(ailoy_vm_obj PUBLIC DMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>)
endif()

# faiss
if(AILOY_WITH_FAISS)
    find_package(OpenMP)
    if(NOT OpenMP_FOUND)
        message(FATAL_ERROR "OpenMP not found. You have to specify OpenMP Root directory by either -DOpenMP_ROOT option or `OpenMP_ROOT` environment variable")
    endif()
    FetchContent_Declare(
        faiss
        GIT_REPOSITORY https://github.com/facebookresearch/faiss.git
        GIT_TAG v1.10.0
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
    target_link_libraries(ailoy_vm_obj PUBLIC faiss)
endif()

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
    
    add_executable(test_tvm_language_model ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_tvm_language_model.cpp)
    add_test(NAME TestTVMLanguageModel COMMAND test_tvm_language_model)
    target_include_directories(test_tvm_language_model PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(test_tvm_language_model PRIVATE ailoy_vm_obj ailoy_broker_obj ailoy_broker_client_obj ailoy_core_obj httplib GTest::gtest)
    target_link_options(test_tvm_language_model PRIVATE -fsanitize=undefined -fsanitize=address)

    add_executable(test_tvm_embedding_model ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_tvm_embedding_model.cpp)
    add_test(NAME TestTVMEmbeddingModel COMMAND test_tvm_embedding_model)
    target_include_directories(test_tvm_embedding_model PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(test_tvm_embedding_model PRIVATE ailoy_vm_obj ailoy_broker_obj ailoy_broker_client_obj ailoy_core_obj GTest::gtest)
    target_link_options(test_tvm_embedding_model PRIVATE -fsanitize=undefined)

    add_executable(test_faiss_vector_store ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_faiss_vector_store.cpp)
    add_test(NAME TestFaissVectorStore COMMAND test_faiss_vector_store)
    target_include_directories(test_faiss_vector_store PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(test_faiss_vector_store PRIVATE ailoy_vm_obj ailoy_broker_obj ailoy_broker_client_obj ailoy_core_obj GTest::gtest)
    target_link_options(test_faiss_vector_store PRIVATE -fsanitize=undefined -fsanitize=address)

    add_executable(test_chromadb_vector_store ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_chromadb_vector_store.cpp)
    add_test(NAME TestChromadbVectorStore COMMAND test_chromadb_vector_store)
    target_include_directories(test_chromadb_vector_store PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(test_chromadb_vector_store PRIVATE ailoy_vm_obj ailoy_broker_obj ailoy_broker_client_obj ailoy_core_obj httplib GTest::gtest)
    target_link_options(test_chromadb_vector_store PRIVATE -fsanitize=undefined -fsanitize=address)

    add_executable(test_split_text ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_split_text.cpp)
    add_test(NAME TestSplitText COMMAND test_split_text)
    target_link_libraries(test_split_text PRIVATE ailoy_vm_obj ailoy_broker_obj ailoy_broker_client_obj ailoy_core_obj httplib GTest::gtest)
    target_link_options(test_split_text PRIVATE -fsanitize=undefined -fsanitize=address)

    add_executable(test_calculator ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_calculator.cpp)
    add_test(NAME TestCalculator COMMAND test_calculator)
    target_link_libraries(test_calculator PRIVATE ailoy_vm_obj ailoy_broker_obj ailoy_broker_client_obj ailoy_core_obj tinyexpr GTest::gtest)
    target_link_options(test_calculator PRIVATE -fsanitize=undefined -fsanitize=address)

    add_executable(test_model_cache ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_model_cache.cpp)
    add_test(NAME TestModelCache COMMAND test_model_cache)
    target_include_directories(test_model_cache PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(test_model_cache PRIVATE ailoy_vm_obj ailoy_broker_obj ailoy_broker_client_obj ailoy_core_obj httplib GTest::gtest)
    target_link_options(test_model_cache PRIVATE -fsanitize=undefined -fsanitize=address)

    add_executable(test_http_request ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_http_request.cpp)
    add_test(NAME TestHttpRequest COMMAND test_http_request)
    target_include_directories(test_http_request PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(test_http_request PRIVATE ailoy_vm_obj ailoy_broker_obj ailoy_broker_client_obj ailoy_core_obj httplib nlohmann_json::nlohmann_json GTest::gtest)
    target_link_options(test_http_request PRIVATE -fsanitize=undefined -fsanitize=address)

    add_executable(test_openai ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_openai.cpp)
    add_test(NAME TestOpenAI COMMAND test_openai)
    target_include_directories(test_openai PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(test_openai PRIVATE ailoy_vm_obj ailoy_broker_obj ailoy_broker_client_obj ailoy_core_obj httplib nlohmann_json::nlohmann_json GTest::gtest GTest::gmock)
    target_link_options(test_openai PRIVATE -fsanitize=undefined -fsanitize=address)

endif()
