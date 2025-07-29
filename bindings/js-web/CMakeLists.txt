if(DEFINED EMSCRIPTEN)
    file(GLOB_RECURSE AILOY_JS_WEB_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/csrc/*.cpp)
    message(STATUS "AILOY_JS_WEB_SRCS: ${AILOY_JS_WEB_SRCS}")
    add_executable(ailoy_js_web ${AILOY_JS_WEB_SRCS})
    target_link_libraries(ailoy_js_web PUBLIC  ailoy_core_obj ailoy_broker_client_obj ailoy_broker_obj ailoy_vm_obj)
    target_link_libraries(ailoy_js_web PRIVATE embind)
    set_target_properties(ailoy_js_web PROPERTIES
        LINK_FLAGS "-sASSERTIONS=1 \
                    -sNO_DISABLE_EXCEPTION_CATCHING=1 \
                    -sWASM_BIGINT=1 \
                    -sPTHREAD_POOL_SIZE=8 \
                    -sINITIAL_MEMORY=134217728 \
                    -sALLOW_MEMORY_GROWTH=1 \
                    -sMODULARIZE=1 \
                    -sEXPORT_ES6=1 \
                    -sENVIRONMENT='worker' \
                    -sUSE_PTHREADS=1 \
                    --emit-tsd=ailoy_js_web.d.ts \
                    --no-entry"
    )
    target_link_options(ailoy_js_web PRIVATE
        "SHELL:-Wl,--allow-multiple-definition"
    )

    add_custom_command(
        TARGET ailoy_js_web
        POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy
            "$<TARGET_FILE_DIR:ailoy_js_web>/ailoy_js_web.wasm"
            "$<TARGET_FILE_DIR:ailoy_js_web>/ailoy_js_web.js"
            "$<TARGET_FILE_DIR:ailoy_js_web>/ailoy_js_web.d.ts"
            "${CMAKE_CURRENT_SOURCE_DIR}/src"
        COMMENT "Copying to output directory"
    )
endif()