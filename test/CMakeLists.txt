file(GLOB TEST_SRCS *.h *.cc)
message(${TEST_SRCS})

get_filename_component(TNN_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..  ABSOLUTE)

include_directories(${TNN_ROOT})

add_executable(TNNTest ${TEST_SRCS})

target_link_libraries(TNNTest
    TNN
    gflags
    )

if(TNN_UNIT_TEST_ENABLE)
    add_subdirectory(unit_test)
endif()
