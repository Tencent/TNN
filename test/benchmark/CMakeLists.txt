
file(GLOB BENCHMARK_SRCS *.cc layer_test/*.cc utils/*.cc ../test_utils.cc ../flags.cc)
message(${BENCHMARK_SRCS})
include_directories(${CMAKE_SOURCE_DIR})

add_executable(TNNBench ${BENCHMARK_SRCS})

target_link_libraries(TNNBench
    TNN
    gflags
)
