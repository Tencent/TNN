file(GLOB_RECURSE DIRECTX_SRC *.h *.cc)

add_library(TNNDIRECTX OBJECT ${DIRECTX_SRC})

if (MSVC)
    add_definitions(-DUNICODE -D_UNICODE)
endif()
