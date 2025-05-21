cmake_minimum_required(VERSION 3.15)
project(ailoy C CXX)

# Compatibility with CMake < 3.5 has been removed from CMake>=4.0.
# Add -DCMAKE_POLICY_VERSION_MINIMUM=3.5 to try configuring anyway.
if(CMAKE_VERSION VERSION_GREATER_EQUAL "4.0.0")
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
endif()

include(CMakePackageConfigHelpers)
include(FetchContent)

# Avoid FetchContent warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24:
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
    cmake_policy(SET CMP0135 NEW)
endif()

# Some platform-specific global options
if(APPLE)
    add_compile_definitions(APPLE)
elseif(UNIX)
    add_compile_definitions(UNIX)
elseif(MSVC)
    add_compile_options(/utf-8)
endif()


##################
# Global options #
##################
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(AILOY_WITH_TEST ON CACHE BOOL "Build test")
if(DEFINED EMSCRIPTEN)
    set(AILOY_WITH_MLC_LLM OFF CACHE BOOL "Enable mlc-llm(tvm) support")
    set(AILOY_WITH_FAISS OFF CACHE BOOL "Enable faiss support")
    set(AILOY_USE_DUMMY_OPENMP ON CACHE BOOL "Use dummy OpenMP")
else()
    set(AILOY_WITH_MLC_LLM ON CACHE BOOL "Enable mlc-llm(tvm) support")
    set(AILOY_WITH_FAISS ON CACHE BOOL "Enable faiss support")
    set(AILOY_USE_DUMMY_OPENMP OFF CACHE BOOL "Use dummy openmp")
endif()
set(MLC_LLM_ROOT "" CACHE PATH "Root directory of mlc-llm. If AILOY_WITH_MLC_LLM is ON and this is empty, it will be downloaded automatically.")

if(AILOY_USE_DUMMY_OPENMP)
    list(PREPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/_deps/dummy_openmp/cmake")
endif()
    
FetchContent_Declare(magic_enum URL https://github.com/Neargye/magic_enum/archive/refs/tags/v0.9.7.tar.gz)
FetchContent_MakeAvailable(magic_enum)

if(AILOY_WITH_TEST)
    FetchContent_Declare(googletest URL https://github.com/google/googletest/archive/refs/tags/v1.16.0.tar.gz)
    set(INSTALL_GTEST OFF)
    if(WIN32)
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    endif()
    FetchContent_MakeAvailable(googletest)
endif()

##################
# Subdirectories #
##################
add_subdirectory(core)
add_subdirectory(broker)
add_subdirectory(vm)

# add_subdirectory(bindings/js-node)

add_library(ailoy SHARED)
target_link_libraries(ailoy PRIVATE
    ailoy_core_obj
    ailoy_broker_obj
    ailoy_broker_client_obj
    ailoy_vm_obj
)
set_target_properties(ailoy PROPERTIES
    INSTALL_RPATH "@loader_path"
)

install(TARGETS ailoy
    EXPORT ailoyTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(EXPORT ailoyTargets
    FILE ailoyTargets.cmake
    NAMESPACE ailoy::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ailoy
)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/ailoyConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/ailoyConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ailoy
)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/ailoyConfig.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ailoy
)
