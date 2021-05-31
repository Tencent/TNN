// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#ifndef TNN_SOURCE_TNN_UTILS_OMP_UTILS_H_
#define TNN_SOURCE_TNN_UTILS_OMP_UTILS_H_

#ifdef _OPENMP

#include <omp.h>

#ifdef _MSC_VER
#define PRAGMA_(X) __pragma(X)
#else
#define PRAGMA_(X) _Pragma(#X)
#endif

#if OpenMP_CXX_VERSION_MAJOR >= 3
#define OMP_PARALLEL_FOR_COLLAPSE_(t) PRAGMA_(omp parallel for collapse(t))
#else
#define OMP_PARALLEL_FOR_COLLAPSE_(t) PRAGMA_(omp parallel for)
#endif
#define OMP_PARALLEL_FOR_ PRAGMA_(omp parallel for)
#define OMP_PARALLEL_FOR_GUIDED_ PRAGMA_(omp parallel for)
#define OMP_PARALLEL_FOR_DYNAMIC_ PRAGMA_(omp parallel for schedule(dynamic))
#define OMP_SECTION_ PRAGMA_(omp section)
#define OMP_PARALLEL_SECTIONS_ PRAGMA_(omp parallel sections)
#define OMP_CORES_ (omp_get_num_procs())
#define OMP_MAX_THREADS_NUM_ (omp_get_max_threads())
#define OMP_TID_ (omp_get_thread_num())
#define OMP_SET_THREADS_(t) (omp_set_num_threads(t))

#else

#define OMP_PARALLEL_FOR_
#define OMP_PARALLEL_FOR_GUIDED_
#define OMP_PARALLEL_FOR_DYNAMIC_
#define OMP_PARALLEL_FOR_COLLAPSE_(t)
#define OMP_SECTION_
#define OMP_PARALLEL_SECTIONS_
#define OMP_CORES_ (1)
#define OMP_MAX_THREADS_NUM_ (1)
#define OMP_TID_ (0)
#define OMP_SET_THREADS_(t)

#endif  // _OPENMP
#endif  // TNN_SOURCE_TNN_UTILS_OMP_UTILS_H_
