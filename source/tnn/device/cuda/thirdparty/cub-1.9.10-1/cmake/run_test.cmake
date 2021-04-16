execute_process(
  COMMAND "${CUB_BINARY}"
  RESULT_VARIABLE EXIT_CODE
)

if (NOT "0" STREQUAL "${EXIT_CODE}")
    message(FATAL_ERROR "${CUB_BINARY} failed (${EXIT_CODE})")
endif ()
