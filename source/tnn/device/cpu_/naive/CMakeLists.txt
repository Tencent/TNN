file(GLOB_RECURSE NAIVE_SRC *.h *.cc)

add_library(TNNNaive OBJECT ${NAIVE_SRC})

#if(SYSTEM.Android)
#    target_link_libraries(TNNCpu log)
#endif()
