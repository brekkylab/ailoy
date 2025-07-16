include(FetchContent)
include(ExternalProject)
include(GNUInstallDirs)

# nlohmann_json
FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
    EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(nlohmann_json)

# dlpack
FetchContent_Declare(dlpack URL https://github.com/dmlc/dlpack/archive/refs/tags/v1.1.tar.gz EXCLUDE_FROM_ALL)
set(BUILD_MOCK OFF CACHE BOOL "DLPack disable mock build" FORCE)  # skip building mock in dlpack
FetchContent_MakeAvailable(dlpack)

# stduuid
FetchContent_Declare(stduuid URL https://github.com/mariusbancila/stduuid/archive/refs/tags/v1.2.3.tar.gz EXCLUDE_FROM_ALL)
FetchContent_MakeAvailable(stduuid)

# spdlog
FetchContent_Declare(spdlog URL https://github.com/gabime/spdlog/archive/refs/tags/v1.15.2.tar.gz EXCLUDE_FROM_ALL)
FetchContent_MakeAvailable(spdlog)

if(NOT DEFINED EMSCRIPTEN)
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

    # openssl
    find_package(OpenSSL REQUIRED)

    # cpptrace
    FetchContent_Declare(cpptrace URL https://github.com/jeremy-rifkin/cpptrace/archive/refs/tags/v0.7.5.tar.gz EXCLUDE_FROM_ALL)
    FetchContent_MakeAvailable(cpptrace)
endif()

file(GLOB_RECURSE AILOY_CORE_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
add_library(ailoy_core_obj OBJECT ${AILOY_CORE_SRCS})
target_include_directories(ailoy_core_obj PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(ailoy_core_obj PUBLIC nlohmann_json::nlohmann_json)
target_link_libraries(ailoy_core_obj PUBLIC dlpack::dlpack)
target_link_libraries(ailoy_core_obj PRIVATE stduuid)
target_link_libraries(ailoy_core_obj PRIVATE spdlog)
target_link_libraries(ailoy_core_obj PRIVATE magic_enum)
if(NOT DEFINED EMSCRIPTEN)
    target_link_libraries(ailoy_core_obj PRIVATE httplib)
    target_link_libraries(ailoy_core_obj PRIVATE OpenSSL::Crypto OpenSSL::SSL)
    target_link_libraries(ailoy_core_obj PRIVATE cpptrace::cpptrace)
endif()

if(AILOY_WITH_TEST)
    if(EMSCRIPTEN)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -s ASYNCIFY=1 -s FETCH=1 -s ALLOW_MEMORY_GROWTH=1 -fexceptions -lembind")
        # set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -s EXPORTED_FUNCTIONS=['_run_tests']")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -s EXPORTED_RUNTIME_METHODS=['ccall','cwrap','stringToNewUTF8']")
    endif()

    add_executable(test_http ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_http.cpp)
    target_link_libraries(test_http PRIVATE ailoy_core_obj)
    set_target_properties(test_http PROPERTIES OUTPUT_NAME "test_http")
    if(EMSCRIPTEN)
        configure_file("${CMAKE_CURRENT_SOURCE_DIR}/tests/test_http.html" "${CMAKE_CURRENT_BINARY_DIR}/test_http.html" COPYONLY)
    else()
        target_link_libraries(test_http PRIVATE GTest::gtest)
        target_link_options(test_http PRIVATE -fsanitize=undefined -fsanitize=address)
    endif()

    add_executable(test_filesystem ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_filesystem.cpp)
    target_link_libraries(test_filesystem PRIVATE ailoy_core_obj)
    set_target_properties(test_filesystem PROPERTIES OUTPUT_NAME "test_filesystem")
    if(EMSCRIPTEN)
        configure_file("${CMAKE_CURRENT_SOURCE_DIR}/tests/test_filesystem.html" "${CMAKE_CURRENT_BINARY_DIR}/test_filesystem.html" COPYONLY)
    endif()

    if(NOT EMSCRIPTEN)
        add_executable(test_packet ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_packet.cpp)
        add_test(NAME TestPacket COMMAND test_packet)
        target_link_libraries(test_packet PRIVATE ailoy_core_obj GTest::gtest)
        target_link_options(test_packet PRIVATE -fsanitize=undefined -fsanitize=address)

        add_executable(test_inproc_socket ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_inproc_socket.cpp)
        add_test(NAME TestInprocSocket COMMAND test_inproc_socket)
        target_link_libraries(test_inproc_socket PRIVATE ailoy_core_obj GTest::gtest)
        target_link_options(test_inproc_socket PRIVATE -fsanitize=undefined -fsanitize=address)

        add_executable(test_value ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_value.cpp)
        target_link_libraries(test_value ailoy_core_obj GTest::gtest)
        add_test(NAME TestValue COMMAND test_value)
        target_link_options(test_value PRIVATE -fsanitize=undefined -fsanitize=address)
    endif()
endif()
