file(GLOB_RECURSE CPU_SRC *.h *.cc)

add_library(TNNCpu OBJECT ${CPU_SRC})

#if(SYSTEM.Android)
#    target_link_libraries(TNNCpu log)
#endif()
