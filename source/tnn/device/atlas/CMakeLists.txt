aux_source_directory(${CMAKE_SOURCE_DIR}/source/tnn/device/atlas ATLAS_SRC)

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/source/tnn/device/atlas/cmake)

find_package(DDK REQUIRED)
include_directories(${DDK_INCLUDE_DIRS} ${DDK_CEREAL_INCLUDE_DIRS} ${DDK_PROTOBUF_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/source/tnn/device/atlas)

add_library(TNNAtlas OBJECT ${ATLAS_SRC})

add_subdirectory(${CMAKE_SOURCE_DIR}/source/tnn/device/atlas/cmake/device)
add_subdirectory(${CMAKE_SOURCE_DIR}/source/tnn/device/atlas/cmake/host)
