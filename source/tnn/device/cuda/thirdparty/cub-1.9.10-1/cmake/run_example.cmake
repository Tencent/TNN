include("${CUB_SOURCE}/cmake/common_variables.cmake")

execute_process(
  COMMAND "${CUB_BINARY}"
  ${FILECHECK_COMMAND}
  RESULT_VARIABLE EXIT_CODE
  OUTPUT_VARIABLE STDOUT
  ERROR_VARIABLE STDERR
)

if (NOT "0" STREQUAL "${EXIT_CODE}")
  message(FATAL_ERROR "${CUB_BINARY} failed (${EXIT_CODE}):\n${STDERR}")
endif ()

if (CHECK_EMPTY_OUTPUT)
  string(LENGTH "${OUTPUT_VARIABLE}" LENGTH)
  if (NOT ${LENGTH} EQUAL 0)
    message(FATAL_ERROR "${CUB_BINARY}: output received, but not expected.")
  endif ()
endif ()
