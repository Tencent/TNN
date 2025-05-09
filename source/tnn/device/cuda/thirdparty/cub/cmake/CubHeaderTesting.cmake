# For every public header, build a translation unit containing `#include <header>`
# to let the compiler try to figure out warnings in that header if it is not otherwise
# included in tests, and also to verify if the headers are modular enough.
# .inl files are not globbed for, because they are not supposed to be used as public
# entrypoints.

# Meta target for all configs' header builds:
add_custom_target(cub.all.headers)

function(cub_add_header_test label definitions)
  foreach(cub_target IN LISTS CUB_TARGETS)
    cub_get_target_property(config_dialect ${cub_target} DIALECT)
    cub_get_target_property(config_prefix ${cub_target} PREFIX)

    set(headertest_target ${config_prefix}.headers.${label})

    cccl_generate_header_tests(${headertest_target} cub
      DIALECT ${config_dialect}
      GLOBS "cub/*.cuh"
    )
    target_link_libraries(${headertest_target} PUBLIC ${cub_target})
    target_compile_definitions(${headertest_target} PRIVATE ${definitions})
    cub_clone_target_properties(${headertest_target} ${cub_target})
    cub_configure_cuda_target(${headertest_target} RDC ${CUB_FORCE_RDC})

    add_dependencies(cub.all.headers ${headertest_target})
    add_dependencies(${config_prefix}.all ${headertest_target})
  endforeach()
endfunction()

# Wrap Thrust/CUB in a custom namespace to check proper use of ns macros:
set(header_definitions
  "THRUST_WRAPPED_NAMESPACE=wrapped_thrust"
  "CUB_WRAPPED_NAMESPACE=wrapped_cub")
cub_add_header_test(base "${header_definitions}")

# Check that BF16 support can be disabled
set(header_definitions
  "THRUST_WRAPPED_NAMESPACE=wrapped_thrust"
  "CUB_WRAPPED_NAMESPACE=wrapped_cub"
  "CCCL_DISABLE_BF16_SUPPORT")
cub_add_header_test(no_bf16 "${header_definitions}")

# Check that half support can be disabled
set(header_definitions
  "THRUST_WRAPPED_NAMESPACE=wrapped_thrust"
  "CUB_WRAPPED_NAMESPACE=wrapped_cub"
  "CCCL_DISABLE_FP16_SUPPORT")
cub_add_header_test(no_half "${header_definitions}")
