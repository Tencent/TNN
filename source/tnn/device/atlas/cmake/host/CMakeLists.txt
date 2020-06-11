set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/source/tnn/device/atlas/cmake)

find_package(DDK REQUIRED)
include_directories(${DDK_INCLUDE_DIRS} ${DDK_CEREAL_INCLUDE_DIRS} ${DDK_PROTOBUF_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/source/tnn/device/atlas)
include_directories(${CMAKE_SOURCE_DIR}/source/tnn/device/atlas/engines)

set(ATLAS_ENGINE_COMMON_SRC
    ${CMAKE_SOURCE_DIR}/source/tnn/device/atlas/atlas_common_types.cc
    ${CMAKE_SOURCE_DIR}/source/tnn/device/atlas/atlas_utils.cc
    )

aux_source_directory(${CMAKE_SOURCE_DIR}/source/tnn/device/atlas/engines/output_engine ATLAS_OUTPUT_ENGINE_SRC)
add_library(OutputEngine SHARED ${ATLAS_OUTPUT_ENGINE_SRC} ${ATLAS_ENGINE_COMMON_SRC})
target_include_directories(OutputEngine PUBLIC ${CMAKE_SOURCE_DIR}/source/tnn/device/atlas/engines/output_engine)
target_link_libraries(OutputEngine PUBLIC ${DDK_HOST_LIBRARIES})
set_target_properties(OutputEngine PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
