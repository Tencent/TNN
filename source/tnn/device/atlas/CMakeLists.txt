aux_source_directory(${CMAKE_SOURCE_DIR}/source/tnn/device/atlas ATLAS_SRC)

include_directories(${CMAKE_SOURCE_DIR}/source/tnn/device/atlas)

set(ASCEND_PATH $ENV{DDK_PATH})

if (NOT DEFINED ENV{DDK_PATH})
    set(ASCEND_PATH "/usr/local/Ascend")
    message(STATUS "set default ASCEND_PATH: ${ASCEND_PATH}")
else ()
    message(STATUS "env ASCEND_PATH: ${ASCEND_PATH}")
endif()

set(ACL_LIB_PATH $ENV{NPU_HOST_LIB})

if (NOT DEFINED ENV{NPU_HOST_LIB})
    set(ACL_LIB_PATH "/usr/local/Ascend/acllib/lib64/stub/")
    message(STATUS "set default ACL_LIB_PATH: ${ACL_LIB_PATH}")
else ()
    message(STATUS "env ACL_LIB_PATH: ${ACL_LIB_PATH}")
endif()

add_definitions(-DENABLE_DVPP_INTERFACE)

# Header path
include_directories(
    ${ASCEND_PATH}/acllib/include/
)

# add host lib path
link_directories(
    ${ACL_LIB_PATH}
)

add_library(TNNAtlas OBJECT ${ATLAS_SRC})
