file(GLOB_RECURSE X86_SRC *.h *.cc)

add_library(TNNX86 OBJECT ${X86_SRC})