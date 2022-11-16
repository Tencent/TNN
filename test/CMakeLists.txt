file(GLOB TEST_SRCS *.h *.cc)
#message(${TEST_SRCS})

if(UNIX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=default")
endif()

get_filename_component(TNN_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..  ABSOLUTE)

include_directories(${TNN_ROOT})

add_executable(TNNTest ${TEST_SRCS})

if(TNN_BUILD_SHARED)
    target_link_libraries(TNNTest
        TNN
        gflags
    )

    if(TNN_BUILD_BENCHMARK_TEST_LIB_ENABLE)
        add_library(TNNBenchmarkTest SHARED ${TEST_SRCS})

        target_link_libraries(TNNBenchmarkTest
            TNN
            gflags
        )
    endif()
elseif(SYSTEM.iOS OR SYSTEM.Darwin)
    target_link_libraries(TNNTest
        -Wl,-force_load TNN
        gflags
    )

    if(TNN_BUILD_BENCHMARK_TEST_LIB_ENABLE)
        add_library(TNNBenchmarkTest STATIC ${TEST_SRCS})
    endif()
else()
    message("target link libraries whole-archive")
    target_link_libraries(TNNTest
        -Wl,--whole-archive TNN -Wl,--no-whole-archive
        gflags
    )

    if(TNN_BUILD_BENCHMARK_TEST_LIB_ENABLE)
        add_library(TNNBenchmarkTest STATIC ${TEST_SRCS})
    endif()
endif()

if(TNN_TENSORRT_ENABLE)
    target_link_libraries(TNNTest nvinfer)
endif()

if(TNN_UNIT_TEST_ENABLE)
    add_subdirectory(unit_test)
endif()

if(TNN_MATCHER_TEST_ENABLE)
    add_subdirectory(matcher_test)
endif()
