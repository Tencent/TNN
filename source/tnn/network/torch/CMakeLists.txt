file(GLOB TNN_TORCH_SRC *.h *.cc)

set(CMAKE_CXX_STANDARD 14)

if(NOT DEFINED ENV{LIBTORCH_ROOT_DIR})
    message(FATAL_ERROR "not defined environment variable:LIBTORCH_ROOT_DIR")
endif()
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} $ENV{LIBTORCH_ROOT_DIR})
find_package(Torch REQUIRED)
message(${TORCH_INCLUDE_DIRS})
include_directories(${TORCH_INCLUDE_DIRS})

if(TNN_TORCHVISION_ENABLE)
    if(NOT DEFINED ENV{LIBTORCHVISION_ROOT_DIR})
        message(FATAL_ERROR "not defined environment variable:LIBTORCHVISION_ROOT_DIR")
    endif()
    set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} $ENV{LIBTORCHVISION_ROOT_DIR})
    find_package(TorchVision REQUIRED)
    include_directories(${TorchVision_INCLUDE_DIR})
endif()

if (NOT TNN_TIACC_MODE)
    add_definitions(-DSAVE_CACHE_FILE)
endif()

enable_language(CUDA)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

message(${TNN_TORCH_SRC})
add_library(TNNTorch OBJECT ${TNN_TORCH_SRC})
