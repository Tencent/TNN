# This file defines the `cub_build_compiler_targets()` function, which
# creates the following interface targets:
#
# cub.compiler_interface
# - Interface target linked into all targets in the CUB developer build.
#   This should not be directly used; it is only used to construct the
#   per-dialect targets below.
#
# cub.compiler_interface_cppXX
# - Interface targets providing dialect-specific compiler flags. These should
#   be linked into the developer build targets, as they include both
#   cub.compiler_interface and cccl.compiler_interface_cppXX.

function(cub_build_compiler_targets)
  set(cuda_compile_options)
  set(cxx_compile_options)
  set(cxx_compile_definitions)

  cccl_build_compiler_interface(cub.compiler_interface
    "${cuda_compile_options}"
    "${cxx_compile_options}"
    "${cxx_compile_definitions}"
  )

  foreach (dialect IN LISTS CCCL_KNOWN_CXX_DIALECTS)
    add_library(cub.compiler_interface_cpp${dialect} INTERFACE)
    target_link_libraries(cub.compiler_interface_cpp${dialect} INTERFACE
      # order matters here, we need the project options to override the cccl options.
      cccl.compiler_interface_cpp${dialect}
      cub.compiler_interface
    )
  endforeach()
endfunction()
