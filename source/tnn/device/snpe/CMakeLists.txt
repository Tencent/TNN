file(GLOB_RECURSE SNPE_SRC *.h *.cc)

add_library(TNNSNPE OBJECT ${SNPE_SRC})

# Verson 2.9 
include_directories(../../../../third_party/snpe/include/zdl)
# Version 2.11+
include_directories(../../../../third_party/snpe/include/SNPE)
