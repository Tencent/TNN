add_subdirectory(third_party/googletest)

set(GTEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/googletest")
set(GTEST_INCLUDE_DIR "${GTEST_DIR}/include")
include_directories(${GTEST_INCLUDE_DIR})

if(TNN_UNIT_TEST_BENCHMARK)
    add_definitions(-DTNN_UNIT_TEST_BENCHMARK)
endif()

file(GLOB UNIT_TEST_SRCS *.cc layer_test/*.cc utils/*.cc ../test_utils.cc ../flags.cc ../timer.cc)
#message(${UNIT_TEST_SRCS})
include_directories(${CMAKE_SOURCE_DIR}/test/unit_test)
include_directories(${CMAKE_SOURCE_DIR})

add_executable(unit_test ${UNIT_TEST_SRCS})

target_link_libraries(unit_test
    TNN
    gtest
    gflags
    )

if(TNN_TENSORRT_ENABLE)
    target_link_libraries(unit_test nvinfer)
endif()

add_test(NAME unit_test COMMAND unit_test)
