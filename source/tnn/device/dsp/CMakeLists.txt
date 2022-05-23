file(GLOB_RECURSE DSP_SRC *.h *.cc)

add_library(TNNDSP OBJECT ${DSP_SRC})

include_directories(thirdparty/snpe/include/zdl)
