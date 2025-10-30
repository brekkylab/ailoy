include(FetchContent)

file(GLOB_RECURSE AILOY_BROKER_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/broker.cpp)
add_library(ailoy_broker_obj OBJECT ${AILOY_BROKER_SRCS})
target_include_directories(ailoy_broker_obj PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(ailoy_broker_obj PUBLIC ailoy_core_obj)
target_link_libraries(ailoy_broker_obj PRIVATE magic_enum)

file(GLOB_RECURSE AILOY_BROKER_CLIENT_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/broker_client.cpp)
add_library(ailoy_broker_client_obj OBJECT ${AILOY_BROKER_CLIENT_SRCS})
target_include_directories(ailoy_broker_client_obj PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(ailoy_broker_client_obj PUBLIC ailoy_core_obj)

if(AILOY_WITH_TEST)
    add_executable(test_broker ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_broker.cpp)
    add_test(NAME TestBroker COMMAND test_broker)
    target_link_libraries(test_broker PRIVATE ailoy_core_obj ailoy_broker_client_obj ailoy_broker_obj GTest::gtest)
    target_link_options(test_broker PRIVATE -fsanitize=undefined)
endif()
