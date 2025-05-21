if(DEFINED SKBUILD)
    FetchContent_Declare(pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v2.13.6
        EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(pybind11)

    file(GLOB_RECURSE AILOY_PY_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/csrc/*.cpp)
    pybind11_add_module(ailoy_py ${AILOY_PY_SRCS})
    target_link_libraries(ailoy_py PUBLIC ailoy_core_obj ailoy_broker_client_obj ailoy_broker_obj ailoy_vm_obj)

    if(APPLE)
        # If using Homebrew LLVM, explicitly tell the compiler to use the macOS system SDK.
        execute_process(
            COMMAND xcrun --sdk macosx --show-sdk-path
            OUTPUT_VARIABLE MACOS_SDK_PATH
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        target_compile_options(ailoy_py PUBLIC -isysroot ${MACOS_SDK_PATH})
        target_link_options(ailoy_py PUBLIC -isysroot ${MACOS_SDK_PATH})
    endif()
    if(MSVC)
        target_compile_definitions(ailoy_py PRIVATE NOMINMAX)
    endif()

    add_custom_command(
        TARGET ailoy_py
        POST_BUILD 
        COMMAND "${CMAKE_COMMAND}" -E copy 
            "$<TARGET_FILE:ailoy_py>"
            "${CMAKE_CURRENT_SOURCE_DIR}/ailoy" 
        COMMENT "Copying to output directory"
    )
endif()
