cmake_minimum_required(VERSION 3.1)
project(TNN-demo)
message(${CMAKE_SOURCE_DIR})
message(${TNN_LIB_PATH})

if(NOT DEFINED ENV{TENSORRT_ROOT_DIR})
    message(FATAL_ERROR "not defined environment variable:TENSORRT_ROOT_DIR")
endif()

if(NOT DEFINED ENV{CUDNN_ROOT_DIR})
    message(FATAL_ERROR "not defined environment variable:CUDNN_ROOT_DIR")
endif()

find_package(CUDA REQUIRED)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_CXX_STANDARD 11)
set(TNN_CUDA_ENABLE ON)
set(TNN_TENSORRT_ENABLE ON)

if (TNN_DEMO_WITH_OPENCV)
    find_package(OpenCV 4 REQUIRED)
    include_directories(${OpenCV_INCLUDE_DIRS})
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -std=c++11 -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11 -pthread -fPIC")

set(TNNRoot ${CMAKE_SOURCE_DIR}/../../../)
set(TNNInclude ${TNNRoot}/include)

include_directories(${TNNInclude})
include_directories(${CMAKE_SOURCE_DIR}/../../base)
include_directories(${CMAKE_SOURCE_DIR}/../../utils)
include_directories(${CUDA_INCLUDE_DIRS})

link_directories(${TNN_LIB_PATH})
link_directories($ENV{TENSORRT_ROOT_DIR}/lib)
link_directories($ENV{CUDNN_ROOT_DIR}/lib)
set(CUDA_LINK_LIBS ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
link_libraries(TNN)
link_libraries(${CUDA_LINK_LIBS})

add_subdirectory(${TNNRoot}/third_party/gflags ${TNNRoot}/third_party/gflags)
get_target_property(GFLAGS_INCLUDE_DIRS gflags INTERFACE_INCLUDE_DIRECTORIES)
include_directories(BEFORE "${GFLAGS_INCLUDE_DIRS}")
link_libraries(gflags)

file(GLOB FLAG_SRC "${CMAKE_SOURCE_DIR}/../src/*.cc")

file(GLOB_RECURSE BASE_SRC
                    "${CMAKE_SOURCE_DIR}/../../base/*.cc")

file(GLOB_RECURSE UTIL_SRC
                   "${CMAKE_SOURCE_DIR}/../../utils/*.cc")

file(GLOB_RECURSE WEBCAM_SRC "${CMAKE_SOURCE_DIR}/../../base/ocr*.cc")
list(REMOVE_ITEM BASE_SRC ${WEBCAM_SRC})

add_definitions(-D_CUDA_)
if (TNN_DEMO_WITH_OPENCV)
    add_definitions(-D_OPENCV_)
endif()
add_executable(demo_cuda_nanodet ${CMAKE_SOURCE_DIR}/../../linux/src/TNNNanodetDetector/TNNNanodetDetector.cc ${BASE_SRC} ${UTIL_SRC} ${FLAG_SRC})

add_executable(demo_cuda_imageclassify ../../linux/src/TNNImageClassify/TNNImageClassify.cc ${BASE_SRC} ${UTIL_SRC} ${FLAG_SRC})
add_executable(demo_cuda_facedetector ../../linux/src/TNNFaceDetector/TNNFaceDetector.cc ${BASE_SRC} ${UTIL_SRC} ${FLAG_SRC})
add_executable(demo_cuda_readingcomprehension ../../linux/src/BertReadingComprehension/BertReadingComprehension.cc ${BASE_SRC} ${UTIL_SRC} ${FLAG_SRC})
add_executable(demo_cuda_objectdetector ../../linux/src/TNNObjectDetector/TNNObjectDetector.cc ${BASE_SRC} ${UTIL_SRC} ${FLAG_SRC})
add_executable(demo_cuda_posedetector ../../linux/src/TNNSkeletonDetector/TNNSkeletonDetector.cc ${BASE_SRC} ${UTIL_SRC} ${FLAG_SRC})
add_executable(demo_cuda_facealignment ../../linux/src/TNNFaceAligner/TNNFaceAligner.cc ${BASE_SRC} ${UTIL_SRC} ${FLAG_SRC})
add_executable(demo_cuda_blazepose ../../linux/src/TNNBlazePose/TNNBlazePose.cc ${BASE_SRC} ${UTIL_SRC} ${FLAG_SRC})

if (TNN_DEMO_WITH_OPENCV) 
    file(GLOB_RECURSE SRC "${CMAKE_SOURCE_DIR}/../../linux/src/TNNWebCamBasedDemo/*.cc")
    add_executable(demo_cuda_webcam ${SRC} ${BASE_SRC} ${UTIL_SRC} ${WEBCAM_SRC})
    add_executable(demo_cuda_ocrdetector ../../linux/src/TNNOcrDetector/TNNOcrDetector.cc ${BASE_SRC} ${UTIL_SRC} ${WEBCAM_SRC} ${FLAG_SRC})
    target_link_libraries(demo_cuda_webcam ${OpenCV_LIBS})
    target_link_libraries(demo_cuda_facedetector ${OpenCV_LIBS})
    target_link_libraries(demo_cuda_objectdetector ${OpenCV_LIBS})
    target_link_libraries(demo_cuda_posedetector ${OpenCV_LIBS})
    target_link_libraries(demo_cuda_facealignment ${OpenCV_LIBS})
    target_link_libraries(demo_cuda_ocrdetector ${OpenCV_LIBS})
    target_link_libraries(demo_cuda_blazepose ${OpenCV_LIBS})
endif()
