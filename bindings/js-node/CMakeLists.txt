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

  if(MSVC AND CMAKE_JS_NODELIB_DEF AND CMAKE_JS_NODELIB_TARGET)
    execute_process(COMMAND ${CMAKE_AR} /def:${CMAKE_JS_NODELIB_DEF} /out:${CMAKE_JS_NODELIB_TARGET} ${CMAKE_STATIC_LINKER_FLAGS})
  endif()

  # Copy target file
  install(TARGETS ailoy_addon
    LIBRARY DESTINATION .
    RUNTIME DESTINATION .
    COMPONENT ailoy_addon
  )

  # Copy runtime dependencies
  install(CODE "set(IGNORE_LIST_FILENAME \"libignore_${CMAKE_SYSTEM_NAME}_${CMAKE_SYSTEM_PROCESSOR}.txt\")")
  install(CODE [[
    file(GET_RUNTIME_DEPENDENCIES
      EXECUTABLES "$<TARGET_FILE:ailoy_addon>"
      RESOLVED_DEPENDENCIES_VAR deps
      UNRESOLVED_DEPENDENCIES_VAR missing
    )

    set(ignorelist_filename "${CMAKE_CURRENT_SOURCE_DIR}/cmake/${IGNORE_LIST_FILENAME}")

    if(EXISTS ${ignorelist_filename})
      message(STATUS "Using ignore file: ${ignorelist_filename}")
      file(STRINGS "${ignorelist_filename}" ignorelist ENCODING UTF-8)
    else()
      message(STATUS "Ignore list not found: ${ignorelist_filename}")
      set(ignorelist "")
    endif()

    foreach(dep IN LISTS deps)
      set(skip FALSE)
      get_filename_component(dep_filename ${dep} NAME)
      foreach(ignore IN LISTS ignorelist)
        if (dep_filename STREQUAL ignore)
          set(skip TRUE)
          break()
        endif()
      endforeach()
      if (NOT skip)
        file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}"
            TYPE SHARED_LIBRARY FILES "${dep}")
      else()
        message(STATUS "Skipping ignored dependency: ${dep}")
      endif()
    endforeach()
]])

# Set RPATH
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(CMAKE_INSTALL_RPATH "\$ORIGIN")
endif()

endif()
