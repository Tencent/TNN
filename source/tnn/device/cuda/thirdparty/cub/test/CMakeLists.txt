if(CMAKE_GENERATOR MATCHES "^Visual Studio")
  if(CUB_ENABLE_RDC_TESTS)
    if("${CMAKE_VERSION}" VERSION_LESS 3.27.5)
      # https://gitlab.kitware.com/cmake/cmake/-/merge_requests/8794
      message(WARNING "CMake 3.27.5 or newer is required to enable RDC tests in Visual Studio.")
      cmake_minimum_required(VERSION 3.27.5)
    endif()
  endif()
endif()

if ("NVHPC" STREQUAL "${CMAKE_CUDA_COMPILER_ID}")
  # NVBugs 200770766
  set(CUB_SEPARATE_CATCH2 ON)
else()
  option(CUB_SEPARATE_CATCH2
    "Build each catch2 test as a separate executable."
    OFF
  )
endif()

cccl_get_c2h()
cccl_get_nvtx()

find_package(CUDAToolkit)

set(build_nvrtc_tests ON)
if ("NVHPC" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
  set(build_nvrtc_tests OFF)
endif()

# The function below reads the filepath `src`, extracts the %PARAM% comments,
# and fills `labels_var` with a list of `label1_value1.label2_value2...`
# strings, and puts the corresponding `DEFINITION=value1:DEFINITION=value2`
# entries into `defs_var`.
#
# See the README.md file in this directory for background info.
function(cub_get_test_params src labels_var defs_var)
  file(READ "${src}" file_data)
  set(param_regex "//[ ]+%PARAM%[ ]+([^ ]+)[ ]+([^ ]+)[ ]+([^\n]*)")

  string(REGEX MATCHALL
    "${param_regex}"
    matches
    "${file_data}"
  )

  set(variant_labels)
  set(variant_defs)

  foreach(match IN LISTS matches)
    string(REGEX MATCH
      "${param_regex}"
      unused
      "${match}"
    )

    set(def ${CMAKE_MATCH_1})
    set(label ${CMAKE_MATCH_2})
    set(values "${CMAKE_MATCH_3}")
    string(REPLACE ":" ";" values "${values}")

    # Build lists of test name suffixes (labels) and preprocessor definitions
    # (defs) containing the cartesian product of all param values:
    if (NOT variant_labels)
      foreach(value IN LISTS values)
        list(APPEND variant_labels ${label}_${value})
      endforeach()
    else()
      set(tmp_labels)
      foreach(old_label IN LISTS variant_labels)
        foreach(value IN LISTS values)
          list(APPEND tmp_labels ${old_label}.${label}_${value})
        endforeach()
      endforeach()
      set(variant_labels "${tmp_labels}")
    endif()

    if (NOT variant_defs)
      foreach(value IN LISTS values)
        list(APPEND variant_defs ${def}=${value})
      endforeach()
    else()
      set(tmp_defs)
      foreach(old_def IN LISTS variant_defs)
        foreach(value IN LISTS values)
          list(APPEND tmp_defs ${old_def}:${def}=${value})
        endforeach()
      endforeach()
      set(variant_defs "${tmp_defs}")
    endif()
  endforeach()

  set(${labels_var} "${variant_labels}" PARENT_SCOPE)
  set(${defs_var} "${variant_defs}" PARENT_SCOPE)
endfunction()

# Create meta targets that build all tests for a single configuration:
foreach(cub_target IN LISTS CUB_TARGETS)
  cub_get_target_property(config_prefix ${cub_target} PREFIX)
  set(config_meta_target ${config_prefix}.tests)
  add_custom_target(${config_meta_target})
  add_dependencies(${config_prefix}.all ${config_meta_target})
endforeach()

file(GLOB test_srcs
  RELATIVE "${CUB_SOURCE_DIR}/test"
  CONFIGURE_DEPENDS
  test_*.cu
  catch2_test_*.cu
)

## _cub_is_catch2_test
#
# If the test_src contains the substring "catch2_test_", `result_var` will
# be set to TRUE.
function(_cub_is_catch2_test result_var test_src)
  string(FIND "${test_src}" "catch2_test_" idx)
  if (idx EQUAL -1)
    set(${result_var} FALSE PARENT_SCOPE)
  else()
    set(${result_var} TRUE PARENT_SCOPE)
  endif()
endfunction()

## _cub_is_fail_test
#
# If the test_src contains the substring "_fail", `result_var` will
# be set to TRUE.
function(_cub_is_fail_test result_var test_src)
  string(FIND "${test_src}" "_fail" idx)
  if (idx EQUAL -1)
    set(${result_var} FALSE PARENT_SCOPE)
  else()
    set(${result_var} TRUE PARENT_SCOPE)
  endif()
endfunction()

## _cub_launcher_requires_rdc
#
# If given launcher id corresponds to a CDP launcher, set `out_var` to 1.
function(_cub_launcher_requires_rdc out_var launcher_id)
  if ("${launcher_id}" STREQUAL "1")
    set(${out_var} 1 PARENT_SCOPE)
  else()
    set(${out_var} 0 PARENT_SCOPE)
  endif()
endfunction()

## cub_add_test
#
# Add a test executable and register it with ctest.
#
# target_name_var: Variable name to overwrite with the name of the test
#   target. Useful for post-processing target information.
# test_name: The name of the test minus "<config_prefix>.test." For example,
#   testing/vector.cu will be "vector", and testing/cuda/copy.cu will be
#   "cuda.copy".
# test_src: The source file that implements the test.
# cub_target: The reference cub target with configuration information.
#
function(cub_add_test target_name_var test_name test_src cub_target launcher_id)
  cub_get_target_property(config_prefix ${cub_target} PREFIX)

  _cub_is_catch2_test(is_catch2_test "${test_src}")
  _cub_launcher_requires_rdc(cdp_val "${launcher_id}")

  # The actual name of the test's target:
  set(test_target ${config_prefix}.test.${test_name})
  set(${target_name_var} ${test_target} PARENT_SCOPE)

  set(config_meta_target ${config_prefix}.tests)

  if (is_catch2_test)
    # Per config helper library:
    set(config_c2h_target ${config_prefix}.test.catch2_helper.lid_${launcher_id})
    if (NOT TARGET ${config_c2h_target})
      add_library(${config_c2h_target} INTERFACE)
      target_include_directories(${config_c2h_target} INTERFACE "${CUB_SOURCE_DIR}/test")
      cub_clone_target_properties(${config_c2h_target} ${cub_target})
      cub_configure_cuda_target(${config_c2h_target} RDC ${cdp_val})
      target_link_libraries(${config_c2h_target} INTERFACE
        ${cub_target}
        cccl.c2h
        CUDA::nvrtc
        CUDA::cuda_driver
      )
    endif() # config_c2h_target

    if (CUB_SEPARATE_CATCH2)
      add_executable(${test_target} "${test_src}")
      target_link_libraries(${test_target} PRIVATE cccl.c2h.main)
      add_dependencies(${config_meta_target} ${test_target})

      add_test(NAME ${test_target} COMMAND "$<TARGET_FILE:${test_target}>")
    else() # Not CUB_SEPARATE_CATCH2
      # Per config catch2 runner
      set(config_c2run_target ${config_prefix}.catch2_test.lid_${launcher_id})
      if (NOT TARGET ${config_c2run_target})
        add_executable(${config_c2run_target})
        target_link_libraries(${config_c2run_target} PRIVATE
          cccl.c2h.main
          ${cub_target}
          ${config_c2h_target}
          Catch2::Catch2)
        cub_clone_target_properties(${config_c2run_target} ${cub_target})
        cub_configure_cuda_target(${config_c2run_target} RDC ${cdp_val})
        add_dependencies(${config_meta_target} ${config_c2run_target})
        target_include_directories(${config_c2run_target} PRIVATE
          "${CUB_SOURCE_DIR}/test"
        )
        if ("NVHPC" STREQUAL "${CMAKE_CUDA_COMPILER_ID}")
          target_link_options(${config_c2run_target} PRIVATE "-cuda")
        endif()

        add_test(NAME ${config_c2run_target}
          COMMAND "$<TARGET_FILE:${config_c2run_target}>"
        )
      endif() # per config catch2 runner

      add_library(${test_target} OBJECT "${test_src}")

      if(CMAKE_GENERATOR MATCHES "^Visual Studio")
        target_link_libraries(${config_c2run_target} PRIVATE $<TARGET_OBJECTS:${test_target}>)
      else()
        target_link_libraries(${config_c2run_target} PRIVATE ${test_target})
      endif()
    endif() # CUB_SEPARATE_CATCH2

    if ("${test_target}" MATCHES "nvrtc")
      target_compile_definitions(${test_target} PRIVATE NVRTC_CUB_PATH="-I${CMAKE_SOURCE_DIR}/cub")
      target_compile_definitions(${test_target} PRIVATE NVRTC_THRUST_PATH="-I${CMAKE_SOURCE_DIR}/thrust")
      target_compile_definitions(${test_target} PRIVATE NVRTC_LIBCUDACXX_PATH="-I${CMAKE_SOURCE_DIR}/libcudacxx/include")
      target_compile_definitions(${test_target} PRIVATE NVRTC_CTK_PATH="-I${CUDAToolkit_INCLUDE_DIRS}")
    endif()

    if ("${test_target}" MATCHES "test.iterator")
      target_compile_options(${test_target} PRIVATE -ftemplate-depth=1000) # for handling large type lists
    endif()

    # enable lambdas for all API examples
    if ("${test_target}" MATCHES "test.[A-Za-z0-9_]+_api")
      target_compile_options(${test_target} PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--extended-lambda>)
    endif()

    target_link_libraries(${test_target} PRIVATE
      ${cub_target}
      ${config_c2h_target}
      Catch2::Catch2
    )
    cub_clone_target_properties(${test_target} ${cub_target})
    target_include_directories(${test_target}
      PUBLIC "${CUB_SOURCE_DIR}/test"
    )
  else() # Not catch2:
    # Related target names:
    set(test_meta_target cub.all.test.${test_name})

    add_executable(${test_target} "${test_src}")
    target_link_libraries(${test_target} PRIVATE
      ${cub_target}
      cccl.c2h
    )
    cub_clone_target_properties(${test_target} ${cub_target})
    target_include_directories(${test_target} PRIVATE "${CUB_SOURCE_DIR}/test")
    target_compile_definitions(${test_target} PRIVATE CUB_DETAIL_DEBUG_ENABLE_SYNC)

    if ("${test_target}" MATCHES "nvtx_in_usercode")
      target_link_libraries(${test_target} PRIVATE nvtx3-cpp)
    endif()

    _cub_is_fail_test(is_fail_test "${test_src}")
    if (is_fail_test)
      set_target_properties(${test_target} PROPERTIES EXCLUDE_FROM_ALL true
                                           EXCLUDE_FROM_DEFAULT_BUILD true)
      add_test(NAME ${test_target}
               COMMAND ${CMAKE_COMMAND} --build "${CMAKE_BINARY_DIR}"
                                        --target ${test_target}
                                        --config $<CONFIGURATION>)
      string(REGEX MATCH "err_([0-9]+)" MATCH_RESULT "${test_name}")
      file(READ ${test_src} test_content)
      if(MATCH_RESULT)
        string(REGEX MATCH "// expected-error-${CMAKE_MATCH_1}+ {{\"([^\"]+)\"}}" expected_errors_matches ${test_content})

        if (expected_errors_matches)
          set_tests_properties(${test_target} PROPERTIES PASS_REGULAR_EXPRESSION "${CMAKE_MATCH_1}")
        else()
          set_tests_properties(${test_target} PROPERTIES WILL_FAIL true)
        endif()
      else()
        string(REGEX MATCH "// expected-error {{\"([^\"]+)\"}}" expected_errors_matches ${test_content})

        if (expected_errors_matches)
          set_tests_properties(${test_target} PROPERTIES PASS_REGULAR_EXPRESSION "${CMAKE_MATCH_1}")
        else()
          set_tests_properties(${test_target} PROPERTIES WILL_FAIL true)
        endif()
      endif()
    else()
      # Add to the active configuration's meta target
      add_dependencies(${config_meta_target} ${test_target})

      # Meta target that builds tests with this name for all configurations:
      if (NOT TARGET ${test_meta_target})
        add_custom_target(${test_meta_target})
      endif()
      add_dependencies(${test_meta_target} ${test_target})

      add_test(NAME ${test_target} COMMAND "$<TARGET_FILE:${test_target}>")
    endif()
  endif() # Not catch2 test
endfunction()

# Sets out_var to launch id if the label contains launch variants
function(_cub_has_lid_variant out_var label)
  string(FIND "${label}" "lid_" idx)
  if (idx EQUAL -1)
    set(${out_var} 0 PARENT_SCOPE)
  else()
    set(${out_var} 1 PARENT_SCOPE)
  endif()
endfunction()

# Sets out_var to 1 if the label contains "lid_1", e.g. launch id corresponds
# to device-side (CDP) launch.
function(_cub_launcher_id out_var label)
  string(REGEX MATCH "lid_([0-9]+)" MATCH_RESULT "${label}")
  if(MATCH_RESULT)
    set(${out_var} ${CMAKE_MATCH_1} PARENT_SCOPE)
  else()
    set(${out_var} 0 PARENT_SCOPE)
  endif()
endfunction()

foreach (test_src IN LISTS test_srcs)
  get_filename_component(test_name "${test_src}" NAME_WE)
  string(REGEX REPLACE "^catch2_test_" "" test_name "${test_name}")
  string(REGEX REPLACE "^test_" "" test_name "${test_name}")

  cub_get_test_params("${test_src}" variant_labels variant_defs)
  list(LENGTH variant_labels num_variants)

  if ("${test_name}" MATCHES "nvrtc")
    if (NOT build_nvrtc_tests)
      continue()
    endif()
  endif()

  # Subtract 1 to support the inclusive endpoint of foreach(...RANGE...):
  math(EXPR range_end "${num_variants} - 1")

  # Verbose output:
  if (num_variants GREATER 0)
    message(VERBOSE "Detected ${num_variants} variants of test '${test_src}':")
    foreach(var_idx RANGE ${range_end})
      math(EXPR i "${var_idx} + 1")
      list(GET variant_labels ${var_idx} label)
      list(GET variant_defs ${var_idx} defs)
      message(VERBOSE "  ${i}: ${test_name} ${label} ${defs}")
    endforeach()
  endif()

  foreach(cub_target IN LISTS CUB_TARGETS)
    cub_get_target_property(config_prefix ${cub_target} PREFIX)

    if (num_variants EQUAL 0)
      if (${CUB_FORCE_RDC})
        set(launcher 1)
      else()
        set(launcher 0)
      endif()

      # Only one version of this test.
      cub_add_test(test_target ${test_name} "${test_src}" ${cub_target} ${launcher})
      cub_configure_cuda_target(${test_target} RDC ${CUB_FORCE_RDC})
    else() # has variants:
      # Meta target to build all parametrizations of the current test for the
      # current CUB_TARGET config
      set(variant_meta_target ${config_prefix}.test.${test_name}.all)
      if (NOT TARGET ${variant_meta_target})
        add_custom_target(${variant_meta_target})
      endif()

      # Meta target to build all parametrizations of the current test for all
      # CUB_TARGET configs
      set(cub_variant_meta_target cub.all.test.${test_name}.all)
      if (NOT TARGET ${cub_variant_meta_target})
        add_custom_target(${cub_variant_meta_target})
      endif()

      # Generate multiple tests, one per variant.
      # See `cub_get_test_params` for details.
      foreach(var_idx RANGE ${range_end})
        list(GET variant_labels ${var_idx} label)
        list(GET variant_defs ${var_idx} defs)
        string(REPLACE ":" ";" defs "${defs}")
        # A unique index per variant:
        list(APPEND defs VAR_IDX=${var_idx})

        # Check if the test explicitly specifies launcher id:
        _cub_has_lid_variant(explicit_launcher "${label}")
        _cub_launcher_id(explicit_launcher_id "${label}")

        if (${explicit_launcher})
          set(launcher_id "${explicit_launcher_id}")
        else()
          if (${CUB_FORCE_RDC})
            set(launcher_id 1)
          else()
            set(launcher_id 0)
          endif()
        endif()

        _cub_launcher_requires_rdc(cdp_val "${launcher_id}")

        if (cdp_val AND NOT CUB_ENABLE_RDC_TESTS)
          continue()
        endif()

        cub_add_test(test_target ${test_name}.${label} "${test_src}" ${cub_target} ${launcher_id})

        # Enable RDC if the test either:
        # 1. Explicitly requests it (lid_1 label)
        # 2. Does not have an explicit CDP variant (no lid_0, lid_1, or lid_2) but
        #    RDC testing is forced
        #
        # Tests that explicitly request no cdp (lid_0 label) should never enable
        # RDC.
        cub_configure_cuda_target(${test_target} RDC ${cdp_val})
        add_dependencies(${variant_meta_target} ${test_target})
        add_dependencies(${cub_variant_meta_target} ${test_target})
        target_compile_definitions(${test_target} PRIVATE ${defs})
      endforeach() # Variant
    endif() # Has variants
  endforeach() # CUB targets
endforeach() # Source file

add_subdirectory(cmake)
