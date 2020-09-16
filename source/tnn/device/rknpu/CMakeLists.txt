file(GLOB RKNPU_SRC
        *.h
        *.cc
        convert/*.h
        convert/*.cc
        convert/math/*.h
        convert/math/*.cc)
add_library(TNNRKNPU OBJECT ${RKNPU_SRC})
include_directories(../../../../third_party/rknpu/rknpu_ddk/include)
