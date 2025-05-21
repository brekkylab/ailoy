include(FetchContent)
include(ExternalProject)
include(GNUInstallDirs)

FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
    EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(nlohmann_json)

FetchContent_Declare(dlpack URL https://github.com/dmlc/dlpack/archive/refs/tags/v1.0.tar.gz EXCLUDE_FROM_ALL)
set(BUILD_MOCK OFF CACHE BOOL "DLPack disable mock build" FORCE)  # skip building mock in dlpack
FetchContent_MakeAvailable(dlpack)

FetchContent_Declare(stduuid URL https://github.com/mariusbancila/stduuid/archive/refs/tags/v1.2.3.tar.gz EXCLUDE_FROM_ALL)
FetchContent_MakeAvailable(stduuid)

FetchContent_Declare(spdlog URL https://github.com/gabime/spdlog/archive/refs/tags/v1.15.2.tar.gz EXCLUDE_FROM_ALL)
FetchContent_MakeAvailable(spdlog)

if(NOT DEFINED EMSCRIPTEN) 
    FetchContent_Declare(cpptrace URL https://github.com/jeremy-rifkin/cpptrace/archive/refs/tags/v0.7.5.tar.gz EXCLUDE_FROM_ALL)
    FetchContent_MakeAvailable(cpptrace)
endif()

file(GLOB_RECURSE AILOY_CORE_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
add_library(ailoy_core_obj OBJECT ${AILOY_CORE_SRCS})
target_include_directories(ailoy_core_obj PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
if(NOT DEFINED EMSCRIPTEN) 
    target_link_libraries(ailoy_core_obj PRIVATE cpptrace::cpptrace)
endif()
target_link_libraries(ailoy_core_obj PUBLIC nlohmann_json::nlohmann_json)
target_link_libraries(ailoy_core_obj PUBLIC dlpack::dlpack)
target_link_libraries(ailoy_core_obj PRIVATE stduuid)
target_link_libraries(ailoy_core_obj PRIVATE spdlog)
target_link_libraries(ailoy_core_obj PRIVATE magic_enum)

if(AILOY_WITH_TEST)
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
