file(GLOB TEST_SRCS *.h *.cc)
#message(${TEST_SRCS})

get_filename_component(TNN_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..  ABSOLUTE)

include_directories(${TNN_ROOT})

add_executable(TNNTest ${TEST_SRCS})

if(TNN_BUILD_SHARED)
    target_link_libraries(TNNTest
        TNN
        gflags
    )
elseif(SYSTEM.iOS OR SYSTEM.Darwin)
    target_link_libraries(TNNTest
        -Wl,-force_load TNN
        gflags
    )
else()
    message("target link libraries whole-archive")
    target_link_libraries(TNNTest
        -Wl,--whole-archive TNN -Wl,--no-whole-archive
        gflags
    )
endif()

if(TNN_UNIT_TEST_ENABLE)
    add_subdirectory(unit_test)
endif()
