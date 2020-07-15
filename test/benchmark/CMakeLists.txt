
file(GLOB BENCHMARK_SRCS *.cc layer_test/*.cc utils/*.cc ../test_utils.cc ../flags.cc)
message(${BENCHMARK_SRCS})
include_directories(${CMAKE_SOURCE_DIR})

add_executable(TNNBench ${BENCHMARK_SRCS})

if(TNN_BUILD_SHARED)
    target_link_libraries(TNNBench
        TNN
        gflags
    )
elseif(SYSTEM.iOS OR SYSTEM.Darwin)
    target_link_libraries(TNNBench
        -Wl,-force_load TNN
        gflags
    )
else()
    message("target link libraries whole-archive")
    target_link_libraries(TNNBench
        -Wl,--whole-archive TNN -Wl,--no-whole-archive
        gflags
    )
endif()


