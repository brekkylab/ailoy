file(GLOB_RECURSE AILOY_COMPONENTS_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/csrc/*)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CARGO_CMD cargo build --verbose)
    set(CARGO_TARGET_DIR target/debug)
else()
    if(NOT CMAKE_BUILD_TYPE STREQUAL "Release" AND NOT CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
        message(WARNING "Unsupported build type ${CMAKE_BUILD_TYPE}, running cargo in release mode")
    endif()
    set(CARGO_CMD cargo build --verbose --release)
    set(CARGO_TARGET_DIR target/release)
endif()

add_custom_target(ailoy_components_rs_build ALL
    COMMAND RUSTFLAGS=${RUST_FLAGS} ${CARGO_CMD} --package ailoy_components_rs
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
    COMMENT "Building Rust library: ailoy_components_rs"
)
add_library(ailoy_components_rs STATIC IMPORTED)
set_target_properties(
    ailoy_components_rs PROPERTIES
    IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/../${CARGO_TARGET_DIR}/libailoy_components_rs.a"
)

add_library(ailoy_components STATIC ${AILOY_COMPONENTS_SRCS})
add_dependencies(ailoy_components ailoy_components_rs_build)
target_include_directories(ailoy_components PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(ailoy_components PRIVATE ailoy_components_rs)

target_link_libraries(ailoy_components PRIVATE nlohmann_json::nlohmann_json)
target_compile_definitions(ailoy_components PUBLIC USE_NLOHMANN_JSON)

target_link_libraries(ailoy_components PRIVATE magic_enum)

target_link_libraries(ailoy_components PRIVATE ailoy_core_obj)

add_test(NAME TestRust
    COMMAND cargo test
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

add_executable(test_message ${CMAKE_CURRENT_SOURCE_DIR}/ctests/test_message.cpp)
add_test(NAME TestPacket COMMAND test_message)
target_link_libraries(test_message PRIVATE ailoy_components GTest::gtest nlohmann_json::nlohmann_json)
target_link_options(test_message PRIVATE -fsanitize=undefined -fsanitize=address)