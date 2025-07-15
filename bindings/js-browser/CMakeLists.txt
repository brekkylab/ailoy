if(DEFINED EMSCRIPTEN)
    file(GLOB_RECURSE AILOY_JS_BROWSER_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/csrc/*.cpp)
    add_executable(ailoy_js_browser ${AILOY_JS_BROWSER_SRCS})
    target_link_libraries(ailoy_js_browser PUBLIC ${AILOY_OBJS})
    target_link_libraries(ailoy_js_browser PRIVATE embind)
    # foreach(TARGET IN ITEMS ailoy_js_browser ${AILOY_OBJS})
    # target_compile_options(${TARGET} PRIVATE -pthread)
    # endforeach()
    set_target_properties(ailoy_js_browser PROPERTIES
    LINK_FLAGS "-sASSERTIONS=1 \
                -sNO_DISABLE_EXCEPTION_CATCHING=1 \
                -sWASM_BIGINT=1 \
                -sPTHREAD_POOL_SIZE=8 \
                -sINITIAL_MEMORY=134217728\
                -sALLOW_MEMORY_GROWTH=1\
                -sMODULARIZE=1 \
                -sEXPORT_ES6=1 \
                -sUSE_ES6_IMPORT_META=0 \
                -sENVIRONMENT='worker' \
                -sUSE_PTHREADS=1 \
                --emit-tsd=ailoy_js_browser.d.ts \
                --no-entry"
    )

    add_custom_command(
    TARGET ailoy_js_browser
    POST_BUILD 
    COMMAND "${CMAKE_COMMAND}" -E copy 
        "$<TARGET_FILE_DIR:ailoy_js_browser>/ailoy_js_browser.wasm"
        "$<TARGET_FILE_DIR:ailoy_js_browser>/ailoy_js_browser.js"
        "$<TARGET_FILE_DIR:ailoy_js_browser>/ailoy_js_browser.d.ts"
        "${CMAKE_CURRENT_SOURCE_DIR}/src" 
    COMMENT "Copying to output directory"
    )

endif()
