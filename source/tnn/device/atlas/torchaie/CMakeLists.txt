
aux_source_directory(${CMAKE_SOURCE_DIR}/source/tnn/device/atlas/torchaie TORCHAIE_SRC)

include_directories(${CMAKE_SOURCE_DIR}/source/tnn/device/atlas/torchaie)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} $ENV{LIBTORCH_ROOT_DIR})
find_package(Torch REQUIRED)
message(${TORCH_INCLUDE_DIRS})
include_directories(${TORCH_INCLUDE_DIRS})
include_directories($ENV{TORCHAIE_ROOT_DIR}/include)

link_directories($ENV{TORCHAIE_ROOT_DIR}/lib)

add_library(TNNTorchAie OBJECT ${TORCHAIE_SRC})