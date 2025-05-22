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
    set(IGNORE_LIST_FILENAME "${CMAKE_CURRENT_SOURCE_DIR}/cmake/${IGNORE_LIST_FILENAME}")
    set(entrypoint "$<TARGET_FILE:ailoy_addon>")
    set(to_process ${entrypoint})
    set(all_deps "")
    set(seen_deps "${entrypoint}")

    # Recusively extract dependencies
    while(NOT "${to_process}" STREQUAL "")
      set(new_to_process "")
      foreach(target IN LISTS to_process)
        file(GET_RUNTIME_DEPENDENCIES
          EXECUTABLES "${target}"
          RESOLVED_DEPENDENCIES_VAR deps
          UNRESOLVED_DEPENDENCIES_VAR missing
        )
        foreach(dep IN LISTS deps)
          list(FIND seen_deps "${dep}" already_seen)
          if(already_seen EQUAL -1)
            list(APPEND all_deps "${dep}")
            list(APPEND new_to_process "${dep}")
            list(APPEND seen_deps "${dep}")
          endif()
        endforeach()
      endforeach()
      set(to_process ${new_to_process})
    endwhile()

    list(REMOVE_DUPLICATES all_deps)

    if(EXISTS ${IGNORE_LIST_FILENAME})
      message(STATUS "Using ignore file: ${IGNORE_LIST_FILENAME}")
      file(STRINGS "${IGNORE_LIST_FILENAME}" ignorelist ENCODING UTF-8)
    else()
      message(STATUS "Ignore list not found: ${IGNORE_LIST_FILENAME}")
      set(ignorelist "")
    endif()

    # Copy files
    foreach(dep IN LISTS all_deps)
      # Find in ignorelist
      set(skip FALSE)
      get_filename_component(dep_filename "${dep}" NAME)
      foreach(ignore IN LISTS ignorelist)
        if(dep_filename STREQUAL ignore)
          set(skip TRUE)
          break()
        endif()
      endforeach()

      if(skip)
        # Skip if in ignorelist
        message(STATUS "Skipping ignored dependency: ${dep}")
      else()
        get_filename_component(real_dep "${dep}" REALPATH)
        get_filename_component(dep_basename "${dep}" NAME) # libfoo.so.1
        get_filename_component(real_basename "${real_dep}" NAME) # libfoo.so.1.2.3
        set(target_path "${CMAKE_INSTALL_PREFIX}/${dep_basename}")

        # Copy real file and rename to symlink name
        file(COPY "${real_dep}" DESTINATION "${CMAKE_INSTALL_PREFIX}")
        file(RENAME "${CMAKE_INSTALL_PREFIX}/${real_basename}" "${target_path}")
        message(STATUS "Renamed ${real_basename} -> ${dep_basename}")
      endif()
    endforeach()
]])

# Set RPATH
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(CMAKE_INSTALL_RPATH "\$ORIGIN")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  set(CMAKE_INSTALL_RPATH "@loader_path")
endif()

endif()
