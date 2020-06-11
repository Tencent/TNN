file(GLOB MODEL_CONVERT_SRCS *.cc)

message(${MODEL_CONVERT_SRCS})

include_directories(${CMAKE_SOURCE_DIR}/tools/model_convert)
include_directories(${CMAKE_SOURCE_DIR}/source/tnn/interpreter/tnn)

add_executable(model_convert ${MODEL_CONVERT_SRCS})

if (SYSTEM.Linux)
  if(TNN_BUILD_SHARED)
    target_link_libraries(model_convert TNN)
  else()
    target_link_libraries(model_convert -Wl,--whole-archive TNN -Wl,--no-whole-archive)
  endif()
elseif(SYSTEM.Darwin)
  if(TNN_BUILD_SHARED)
    target_link_libraries(model_convert TNN)
  else()
    target_link_libraries(model_convert -Wl,-force_load TNN)
  endif()
endif()

set_target_properties(model_convert PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
