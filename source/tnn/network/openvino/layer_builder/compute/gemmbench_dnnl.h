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

#include "dnnl_common.h"
int GemmScan(int m, int n, int k);
#endif