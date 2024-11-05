#pragma once

// Include this file at the top of a unit test for CUB device algorithms to check whether any inserted NVTX ranges nest.

#include <cstdio>
#include <cstdlib>

#include <catch2/catch.hpp>

#if defined(__cpp_inline_variables)
inline thread_local bool entered = false;

struct NestedNVTXRangeGuard
{
  NestedNVTXRangeGuard(const char* name)
  {
    UNSCOPED_INFO("Entering NVTX range " << name);
    if (entered)
    {
      FAIL("Nested NVTX range detected");
    }
    entered = true;
  }

  ~NestedNVTXRangeGuard()
  {
    entered = false;
    UNSCOPED_INFO("Leaving NVTX range");
  }
};

#  define CUB_DETAIL_BEFORE_NVTX_RANGE_SCOPE(name)                              \
    ::cuda::std::optional<::NestedNVTXRangeGuard> __cub_nvtx3_reentrency_guard; \
    NV_IF_TARGET(NV_IS_HOST, __cub_nvtx3_reentrency_guard.emplace(name););
#endif // defined(__cpp_inline_variables)
