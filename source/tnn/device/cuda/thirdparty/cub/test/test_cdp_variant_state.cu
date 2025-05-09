/*
 *  Copyright 2022 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include <cub/detail/detect_cuda_runtime.cuh>

#include <cstdlib>

int main()
{
  // This test just checks that RDC is enabled and detected properly when using
  // the %PARAM% system to request CDP support (see the README.md file in
  // this directory).

  // %PARAM% TEST_LAUNCH lid 0:1:2

#ifdef CUB_RDC_ENABLED
  return (TEST_LAUNCH == 1) ? EXIT_SUCCESS : EXIT_FAILURE;
#else
  return (TEST_LAUNCH == 0 || TEST_LAUNCH == 2) ? EXIT_SUCCESS : EXIT_FAILURE;
#endif
}
