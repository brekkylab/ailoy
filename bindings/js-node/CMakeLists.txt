if (DEFINED NODE)
  add_definitions(-DNAPI_VERSION=7)
  # Add node-addon-api include
  execute_process(
    COMMAND node -p "require('node-addon-api').include"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE NODE_ADDON_API_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  string(REGEX REPLACE "\"" "" NODE_ADDON_API_INCLUDE_DIR ${NODE_ADDON_API_INCLUDE_DIR})
  file(GLOB_RECURSE AILOY_JS_NODE_SHIM_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/csrc/*.cpp)
  add_library(ailoy_addon SHARED ${AILOY_JS_NODE_SHIM_SRCS} ${CMAKE_JS_SRC})
  set_target_properties(ailoy_addon PROPERTIES PREFIX "" SUFFIX ".node")
  target_include_directories(ailoy_addon PRIVATE ${CMAKE_JS_INC} ${NODE_ADDON_API_INCLUDE_DIR})
  find_package(OpenSSL REQUIRED)
  target_link_libraries(ailoy_addon PUBLIC ailoy_core_obj ailoy_broker_client_obj ailoy_broker_obj ailoy_vm_obj OpenSSL::Crypto OpenSSL::SSL ${CMAKE_JS_LIB})
  # Options to enable sanitizer (not working on examples though)
  # target_compile_options(ailoy_addon PRIVATE -fsanitize=address -fno-omit-frame-pointer)
  # target_link_options(ailoy_addon PRIVATE -fsanitize=address)

  if(MSVC AND CMAKE_JS_NODELIB_DEF AND CMAKE_JS_NODELIB_TARGET)
    execute_process(COMMAND ${CMAKE_AR} /def:${CMAKE_JS_NODELIB_DEF} /out:${CMAKE_JS_NODELIB_TARGET} ${CMAKE_STATIC_LINKER_FLAGS})
  endif()

  list(APPEND AILOY_JS_NODE_OUTPUTS ailoy_addon.node)
  if(WIN32)
    list(APPEND AILOY_JS_NODE_OUTPUTS mlc_llm.dll)
    list(APPEND AILOY_JS_NODE_OUTPUTS tvm_runtime.dll)
    if(CMAKE_GENERATOR MATCHES "Visual Studio")
      list(APPEND AILOY_JS_NODE_OUTPUTS $<$<CONFIG:Debug>:ailoy_addon.pdb>)
      list(APPEND AILOY_JS_NODE_OUTPUTS $<$<CONFIG:Debug>:mlc_llm.pdb>)
      list(APPEND AILOY_JS_NODE_OUTPUTS $<$<CONFIG:Debug>:tvm_runtime.pdb>)
    elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
      list(APPEND AILOY_JS_NODE_OUTPUTS ailoy_addon.pdb)
      list(APPEND AILOY_JS_NODE_OUTPUTS mlc_llm.pdb)
      list(APPEND AILOY_JS_NODE_OUTPUTS tvm_runtime.pdb)
    endif()
  endif()

  set(AILOY_JS_NODE_OUTPUTS_SRC ${AILOY_JS_NODE_OUTPUTS})
  list(TRANSFORM AILOY_JS_NODE_OUTPUTS_SRC PREPEND $<TARGET_FILE_DIR:ailoy_addon>/)
  set(AILOY_JS_NODE_OUTPUT_DST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
  set(AILOY_JS_NODE_OUTPUTS_DST ${AILOY_JS_NODE_OUTPUTS})
  list(TRANSFORM AILOY_JS_NODE_OUTPUTS_DST PREPEND ${AILOY_JS_NODE_OUTPUT_DST_DIR}/)
  add_custom_command(
    TARGET ailoy_addon
    POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        ${AILOY_JS_NODE_OUTPUTS_SRC}
        ${AILOY_JS_NODE_OUTPUT_DST_DIR}
    COMMENT "Copying build artifacts to output directory"
    BYPRODUCTS
        ${AILOY_JS_NODE_OUTPUTS_DST}
  )

endif()
