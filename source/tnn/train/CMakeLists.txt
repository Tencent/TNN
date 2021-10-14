file(GLOB_RECURSE TRAIN_SRC *.h *.cc)


add_library(TNNTrain OBJECT ${TRAIN_SRC})

#if(SYSTEM.Android)
#    target_link_libraries(TNNArm log)
#endif()
