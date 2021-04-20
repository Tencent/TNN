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

#ifndef __DNNL_COMMON__
#define __DNNL_COMMON__

#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>
#include <chrono>

#include <dnnl.hpp>
#include <dnnl_debug.h>

using namespace dnnl;

typedef std::unordered_map<std::string, memory*> map_mem_t;
typedef std::unordered_map<std::string, inner_product_forward::primitive_desc*> map_ip_primd_t;
typedef std::unordered_map<std::string, matmul::primitive_desc*> map_mm_primd_t;
typedef std::unordered_map<std::string, primitive*> map_prim_t;

static map_mem_t g_memory;
static map_ip_primd_t g_ip_prim_desc;
static map_mm_primd_t g_mm_prim_desc;
static map_prim_t g_prim;

#endif
