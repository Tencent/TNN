file(GLOB_RECURSE SNPE_SRC *.h *.cc)

add_library(TNNSNPE OBJECT ${SNPE_SRC})

include_directories(../../../../third_party/snpe/include/zdl)
