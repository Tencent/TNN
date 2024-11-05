cccl_get_catch2()
cccl_get_nvbench()

add_library(nvbench_helper OBJECT nvbench_helper/nvbench_helper.cuh
                                  nvbench_helper/nvbench_helper.cu)

target_link_libraries(nvbench_helper PUBLIC CUB::CUB
                                            Thrust::Thrust
                                            CUB::libcudacxx
                                            nvbench::nvbench
                                     PRIVATE CUDA::curand)

target_include_directories(nvbench_helper PUBLIC "${CMAKE_CURRENT_LIST_DIR}/nvbench_helper")
set_target_properties(nvbench_helper PROPERTIES CUDA_STANDARD 17 CXX_STANDARD 17)


option(CUB_ENABLE_NVBENCH_HELPER_TESTS "Enable tests for nvbench_helper" OFF)
mark_as_advanced(CUB_ENABLE_NVBENCH_HELPER_TESTS)

if (CUB_ENABLE_NVBENCH_HELPER_TESTS)
  cccl_get_boost()

  function(add_nvbench_helper_test device_system)
    set(nvbench_helper_test_target nvbench_helper.test.${device_system})
    add_executable(${nvbench_helper_test_target} test/gen_seed.cu
                                                 test/gen_range.cu
                                                 test/gen_entropy.cu
                                                 test/gen_uniform_distribution.cu
                                                 test/gen_power_law_distribution.cu
                                                 test/main.cpp)
    cccl_configure_target(${nvbench_helper_test_target} DIALECT 17)
    target_link_libraries(${nvbench_helper_test_target} PRIVATE nvbench_helper Catch2::Catch2 Boost::math)
    if ("${device_system}" STREQUAL "cpp")
      target_compile_definitions(${nvbench_helper_test_target} PRIVATE THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP)
    endif()
  endfunction()

  add_nvbench_helper_test(cpp)
  add_nvbench_helper_test(cuda)
endif()
