cmake_minimum_required(VERSION 3.1)
project(TNN-demo)
message(${CMAKE_SOURCE_DIR})
message(${TNN_LIB_PATH})

option(TNN_OPENVINO_LIB_PATH "openvino lib path" "")
option(TNN_DEMO_WITH_OPENCV "with opencv" OFF)

set(CMAKE_CXX_STANDARD 11)
set(TNN_CPU_ENABLE ON)
set(TNN_OPENVINO_ENABLE ON)

if (TNN_DEMO_WITH_OPENCV)
    find_package(OpenCV 4 REQUIRED)
    include_directories(${OpenCV_INCLUDE_DIRS})
endif()

if (CMAKE_SYSTEM_NAME MATCHES "Windows")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /w")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /w")
else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -std=c++11 -fPIC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11 -pthread -fPIC")
endif()

set(TNNRoot ${CMAKE_SOURCE_DIR}/../../../)
set(TNNInclude ${TNNRoot}/include)

include_directories(${TNNInclude})
include_directories(${CMAKE_SOURCE_DIR}/../../base)
include_directories(${CMAKE_SOURCE_DIR}/../../utils)
link_directories(${TNN_LIB_PATH})
if (${TNN_OPENVINO_ENABLE})
    link_directories(${TNN_OPENVINO_LIB_PATH})
endif()

link_libraries(TNN)

file(GLOB_RECURSE BASE_SRC
                    "${CMAKE_SOURCE_DIR}/../../base/*.cc")

file(GLOB_RECURSE UTIL_SRC
                   "${CMAKE_SOURCE_DIR}/../../utils/*.cc")

file(GLOB_RECURSE WEBCAM_SRC "${CMAKE_SOURCE_DIR}/../../base/ocr*.cc")
list(REMOVE_ITEM BASE_SRC ${WEBCAM_SRC})

if (TNN_DEMO_WITH_OPENCV)
    add_definitions(-D_OPENCV_)
endif()

add_executable(demo_x86_imageclassify ../src/TNNImageClassify/TNNImageClassify.cc ${BASE_SRC} ${UTIL_SRC})
add_executable(demo_x86_facedetector ../src/TNNFaceDetector/TNNFaceDetector.cc ${BASE_SRC} ${UTIL_SRC})
add_executable(demo_x86_readingcomprehension ../src/BertReadingComprehension/BertReadingComprehension.cc ${BASE_SRC} ${UTIL_SRC})
add_executable(demo_x86_objectdetector ../src/TNNObjectDetector/TNNObjectDetector.cc ${BASE_SRC} ${UTIL_SRC})

if (TNN_DEMO_WITH_OPENCV) 
    file(GLOB_RECURSE SRC "${CMAKE_SOURCE_DIR}/../src/TNNWebCamBasedDemo/*.cc")
    add_executable(demo_x86_webcam ${SRC} ${BASE_SRC} ${UTIL_SRC} ${WEBCAM_SRC})
    target_link_libraries(demo_x86_webcam ${OpenCV_LIBS})
    target_link_libraries(demo_x86_facedetector ${OpenCV_LIBS})
    target_link_libraries(demo_x86_objectdetector ${OpenCV_LIBS})
endif()
