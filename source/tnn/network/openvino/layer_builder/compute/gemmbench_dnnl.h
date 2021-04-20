/*******************************************************************************
* Copyright 2020 Intel Corporation
*
* Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
* in compliance with the License. You may obtain a copy of the License at
* 
* https://opensource.org/licenses/BSD-3-Clause0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/


#ifndef GEMMBENCH_DNNL_
#define GEMMBENCH_DNNL_

#include <iostream>
#include <iomanip>
#include <chrono>

#include <dnnl.hpp>

#include "dnnl_inner_product.h"
#include "dnnl_matmul.h"

int GemmScan(int m, int n, int k);
double test_dnnl_sgemm(float *A, float *B, float *C, int m, int n, int k);
template <typename T_A, typename T_B, typename T_bias, typename T_C>
double test_dnnl_inner_product(engine eng, stream stm, T_A* A_buf, T_B* B_buf, T_bias* bias_buf, T_C* C_buf, int m, int n, int k);
template <typename T_A, typename T_B, typename T_bias, typename T_C>
double test_dnnl_matmul(engine eng, stream stm, T_A* A_buf, T_B* B_buf, T_bias* bias_buf, T_C* C_buf, int m, int n, int k);
#endif