# cub_configure_cuda_target(<target_name> RDC <ON|OFF>)
#
# Configures `target_name` with the appropriate CUDA architectures and RDC state.
function(cub_configure_cuda_target target_name)
  set(options)
  set(one_value_args RDC)
  set(multi_value_args)
  cmake_parse_arguments(cub_cuda "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  if (cub_cuda_UNPARSED_ARGUMENTS)
    message(AUTHOR_WARNING
      "Unrecognized arguments passed to cub_configure_cuda_target: "
      ${cub_cuda_UNPARSED_ARGUMENTS})
  endif()

  if (NOT DEFINED cub_cuda_RDC)
    message(AUTHOR_WARNING "RDC option required for cub_configure_cuda_target.")
  endif()

  if (cub_cuda_RDC)
    set_target_properties(${target_name} PROPERTIES
      CUDA_ARCHITECTURES "${CUB_CUDA_ARCHITECTURES_RDC}"
      POSITION_INDEPENDENT_CODE ON
      CUDA_SEPARABLE_COMPILATION ON)
  else()
    set_target_properties(${target_name} PROPERTIES
      CUDA_ARCHITECTURES "${CUB_CUDA_ARCHITECTURES}"
      CUDA_SEPARABLE_COMPILATION OFF)
  endif()
endfunction()
