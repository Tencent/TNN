# Check source code for issues that can be found by pattern matching:
add_test(
  NAME cub.test.cmake.check_source_files
  COMMAND
    "${CMAKE_COMMAND}"
      -D "CUB_SOURCE_DIR=${CUB_SOURCE_DIR}"
      -P "${CMAKE_CURRENT_LIST_DIR}/check_source_files.cmake"
)
