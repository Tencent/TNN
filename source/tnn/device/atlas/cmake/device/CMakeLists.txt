set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/source/tnn/device/atlas/cmake)
include(${CMAKE_SOURCE_DIR}/source/tnn/device/atlas/cmake/Ascend.cmake)

find_package(DDK REQUIRED)
include_directories(${DDK_INCLUDE_DIRS} ${DDK_CEREAL_INCLUDE_DIRS} ${DDK_PROTOBUF_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/source/tnn/device/atlas)
include_directories(${CMAKE_SOURCE_DIR}/source/tnn/device/atlas/engines)

set(ATLAS_ENGINE_COMMON_SRC
    ${CMAKE_SOURCE_DIR}/source/tnn/device/atlas/engines/atlas_device_utils.cc
    ${CMAKE_SOURCE_DIR}/source/tnn/device/atlas/atlas_common_types.cc
    ${CMAKE_SOURCE_DIR}/source/tnn/device/atlas/atlas_utils.cc
    )

aux_source_directory(${CMAKE_SOURCE_DIR}/source/tnn/device/atlas/engines/inference_engine ATLAS_INFERENCE_ENGINE_SRC)
add_library(InferenceEngine SHARED ${ATLAS_INFERENCE_ENGINE_SRC} ${ATLAS_ENGINE_COMMON_SRC})
target_include_directories(InferenceEngine PUBLIC ${CMAKE_SOURCE_DIR}/source/tnn/device/atlas/engines/inference_engine)
target_link_libraries(InferenceEngine PUBLIC ${DDK_DEVICE_LIBRARIES})
set_target_properties(InferenceEngine PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

aux_source_directory(${CMAKE_SOURCE_DIR}/source/tnn/device/atlas/engines/dvpp_engine ATLAS_DVPP_ENGINE_SRC)
add_library(DvppEngine SHARED ${ATLAS_DVPP_ENGINE_SRC} ${ATLAS_ENGINE_COMMON_SRC})
target_include_directories(DvppEngine PUBLIC ${CMAKE_SOURCE_DIR}/source/tnn/device/atlas/engines/dvpp_engine)
target_link_libraries(DvppEngine PUBLIC ${DDK_DEVICE_LIBRARIES})
set_target_properties(DvppEngine PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
