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
#include "gemm_unit.h"
using namespace dnnl;

// class GemmUnit
// {
// public:
//     GemmUnit(/* args */);
//     ~GemmUnit();

//     template <typename T_input, typename T_wei, typename T_bias, typename T_output>
//     bool InnerProduct(engine eng, stream stm, T_input *input, T_wei *weight, T_bias *bias, T_output *output, int m,
//                       int n, int k) {

//         // generate prim key
//         char type_input   = (std::is_floating_point<T_input>::value) ? 'f' : 'b';
//         char type_weights = (std::is_floating_point<T_wei>::value) ? 'f' : 'b';
//         char type_bias    = (std::is_floating_point<T_bias>::value) ? 'f' : 'b';
//         char type_output  = (std::is_floating_point<T_output>::value) ? 'f' : 'b';

//         const void *address = static_cast<const void *>(weight);

//         std::stringstream weights_addr;
//         weights_addr << "InnerProduct-" << type_input << type_weights << type_bias << type_output << '-' << m << '-'
//                      << n << '-' << k << '-' << address;
//         std::string prim_key = weights_addr.str();

//         // memory disc for primitive
//         memory::dims src_tz     = {m, k};
//         memory::dims weights_tz = {n, k};
//         memory::dims bias_tz    = {n};
//         memory::dims dst_tz     = {m, n};

//         memory::data_type src_dt =
//             (std::is_floating_point<T_input>::value)  ? memory::data_type::f32 : memory::data_type::bf16;
//         memory::data_type weights_dt =
//             (std::is_floating_point<T_wei>::value)    ? memory::data_type::f32 : memory::data_type::bf16;
//         memory::data_type bias_dt =
//             (std::is_floating_point<T_bias>::value)   ? memory::data_type::f32 : memory::data_type::bf16;
//         memory::data_type dst_dt =
//             (std::is_floating_point<T_output>::value) ? memory::data_type::f32 : memory::data_type::bf16;

//         // save created primitive
//         auto it_prim_created = g_prim.find(prim_key);
//         if (it_prim_created == g_prim.end()) {

//             auto src_md     = memory::desc({src_tz}, src_dt, memory::format_tag::any);
//             auto weights_md = memory::desc({weights_tz}, weights_dt, memory::format_tag::any);
//             auto bias_md    = memory::desc({bias_tz}, bias_dt, memory::format_tag::any);
//             auto dst_md     = memory::desc({dst_tz}, dst_dt, memory::format_tag::any);

//             auto desc       = inner_product_forward::desc(prop_kind::forward_inference, src_md, weights_md, bias_md, dst_md);
//             auto prim_desc  = std::make_shared<inner_product_forward::primitive_desc>(desc, eng);
//             auto prim       = std::make_shared<inner_product_forward>(*prim_desc);

//             g_ip_prim_desc.insert(std::pair<std::string, std::shared_ptr<inner_product_forward::primitive_desc>> (prim_key, prim_desc));
//             g_prim.insert(std::pair<std::string, std::shared_ptr<primitive>> (prim_key, prim));

//             std::cout << "InnerProduct: save prim_key = " << prim_key << ", prim number = " << g_prim.size()
//                       << std::endl;
//         }

//         // memory disc for user
//         auto user_src_md     = memory::desc(src_tz, src_dt, memory::format_tag::nc);
//         auto user_weights_md = memory::desc(weights_tz, weights_dt, memory::format_tag::io);  // oi or io
//         auto user_bias_md    = memory::desc(bias_tz, bias_dt, memory::format_tag::x);

//         // create & set memory according to user_disc
//         std::shared_ptr<memory> user_src_memory_ptr     = std::make_shared<dnnl::memory>(user_src_md, eng, input);
//         std::shared_ptr<memory> user_weights_memory_ptr = std::make_shared<dnnl::memory>(user_weights_md, eng, weight);
//         std::shared_ptr<memory> user_bias_memory_ptr    = std::make_shared<dnnl::memory>(user_bias_md, eng, bias);

//         // inner product prim_disc
//         auto prim_desc = g_ip_prim_desc[prim_key];

//         // prepare src_mem
//         auto src_memory_ptr     = user_src_memory_ptr;
//         auto weights_memory_ptr = user_weights_memory_ptr;
//         auto bias_memory_ptr    = user_bias_memory_ptr;

//         if (prim_desc->src_desc() != user_src_memory_ptr->get_desc()) {
//             src_memory_ptr       = std::make_shared<dnnl::memory> (prim_desc->src_desc(), eng);
//             auto reorder_src = reorder(*user_src_memory_ptr, *src_memory_ptr);
//             reorder_src.execute(stm, {{DNNL_ARG_FROM, *user_src_memory_ptr}, 
//                                       {DNNL_ARG_TO, *src_memory_ptr}});
//         }

//         // prepare weights_mem
//         std::string prim_weights_key = prim_key + "-weights";
//         auto it_memory_created = g_memory.find(prim_weights_key);
//         if (it_memory_created == g_memory.end()) {
//             if (prim_desc->weights_desc() != user_weights_memory_ptr->get_desc()) {
//                 weights_memory_ptr       = std::make_shared<memory>(prim_desc->weights_desc(), eng);
//                 auto reorder_weights = reorder(*user_weights_memory_ptr, *weights_memory_ptr);
//                 reorder_weights.execute(stm,
//                                         {{DNNL_ARG_FROM, *user_weights_memory_ptr}, 
//                                          {DNNL_ARG_TO, *weights_memory_ptr}});
//                 stm.wait();
//             }
//             g_memory[prim_weights_key] = weights_memory_ptr;
//         } else {
//             weights_memory_ptr = g_memory[prim_weights_key];
//         }

//         // check if we have saved model weights
//         // if (prim_desc.weights_desc() != user_weights_memory.get_desc()) {
//         //     std::string prim_weights_key = prim_key + "-weights";
//         //     auto it_memory_created       = g_memory.find(prim_weights_key);
//         //     if (it_memory_created == g_memory.end()) {

                
//         //         weights_memory       = new memory(prim_desc.weights_desc(), eng);
//         //         auto reorder_weights = reorder(user_weights_memory, *weights_memory);

//         //         reorder_weights.execute(stm, {{DNNL_ARG_FROM, user_weights_memory}, {DNNL_ARG_TO, *weights_memory}});
//         //         g_memory.insert(std::pair<std::string, memory *>(prim_weights_key, weights_memory));
//         //     } else {
//         //         weights_memory = it_memory_created->second;
//         //     }
//         // }

//         auto bias_mem_key = prim_key + "-Bias";
//         auto bias_mem = g_memory.find(bias_mem_key);
//         if (bias_mem == g_memory.end()) {
//             if (prim_desc->bias_desc() != user_bias_memory_ptr->get_desc()) {
//                     bias_memory_ptr = std::make_shared<memory> (prim_desc->bias_desc(), eng);
//                     auto reorder_bias = reorder(*user_bias_memory_ptr, *bias_memory_ptr);
//                     reorder_bias.execute(stm,
//                                          {{DNNL_ARG_FROM, *user_bias_memory_ptr}, {DNNL_ARG_TO, *bias_memory_ptr}});
//                     stm.wait();
//                     g_memory[bias_mem_key] = bias_memory_ptr;
//             }
//         } else {
//             bias_memory_ptr = bias_mem->second;
//         }



//         // prepare bias_mem
//         // if (prim_desc.bias_desc() != user_bias_memory.get_desc()) {
//         //     std::string prim_bias_key = prim_key + "-bias";
//         //     auto it_memory_created    = g_memory.find(prim_bias_key);
//         //     if (it_memory_created == g_memory.end()) {
//         //         std::cout << "InnerProduct: reorder user_bias_memory !!!" << std::endl;
//         //         bias_memory       = new memory(prim_desc.bias_desc(), eng);
//         //         auto reorder_bias = reorder(user_bias_memory, *bias_memory);
//         //         reorder_bias.execute(stm, {{DNNL_ARG_FROM, user_bias_memory}, {DNNL_ARG_TO, *bias_memory}});
//         //         g_memory.insert(std::pair<std::string, memory *>(prim_bias_key, bias_memory));
//         //     } else {
//         //         bias_memory = it_memory_created->second;
//         //     }
//         // }
        
//         // set dst_memory
//         std::shared_ptr<dnnl::memory> dst_memory_ptr     =std::make_shared<dnnl::memory> (prim_desc->dst_desc(), eng, output);

//         // do inference
//         it_prim_created = g_prim.find(prim_key);
//         it_prim_created->second->execute(stm, {{DNNL_ARG_SRC, *src_memory_ptr},
//                                                {DNNL_ARG_WEIGHTS, *weights_memory_ptr},
//                                                {DNNL_ARG_BIAS, *bias_memory_ptr},
//                                                {DNNL_ARG_DST, *dst_memory_ptr}});

//         stm.wait();
//         return true;
//     }


//     template <typename T_input, typename T_wei, typename T_bias, typename T_output>
//     bool MatMul(engine eng, stream stm, T_input *input, T_wei *weight, T_bias *bias, T_output *output, int m, int n,
//                 int k) {

//         char type_input   = (std::is_floating_point<T_input>::value) ? 'f' : 'b';
//         char type_weights = (std::is_floating_point<T_wei>::value) ? 'f' : 'b';
//         char type_bias    = (std::is_floating_point<T_bias>::value) ? 'f' : 'b';
//         char type_output  = (std::is_floating_point<T_output>::value) ? 'f' : 'b';

//         const void *address = static_cast<const void *>(weight);

//         std::stringstream weights_addr;
//         weights_addr << "MatMul-" << type_input << type_weights << type_bias << type_output << '-' << m << '-' << n
//                      << '-' << k << '-' << address;
//         std::string prim_key = weights_addr.str();

//         memory::dims src_tz     = {m, k};
//         memory::dims weights_tz = {k, n};
//         memory::dims bias_tz    = {1, n};
//         memory::dims dst_tz     = {m, n};

//         memory::data_type src_dt =
//             (std::is_floating_point<T_input>::value) ? memory::data_type::f32 : memory::data_type::bf16;
//         memory::data_type weights_dt =
//             (std::is_floating_point<T_wei>::value) ? memory::data_type::f32 : memory::data_type::bf16;
//         memory::data_type bias_dt =
//             (std::is_floating_point<T_bias>::value) ? memory::data_type::f32 : memory::data_type::bf16;
//         memory::data_type dst_dt =
//             (std::is_floating_point<T_output>::value) ? memory::data_type::f32 : memory::data_type::bf16;

//         auto it_prim_created = g_prim.find(prim_key);
//         if (it_prim_created == g_prim.end()) {

//             auto src_md     = memory::desc({src_tz}, src_dt, memory::format_tag::ab);
//             auto weights_md = memory::desc({weights_tz}, weights_dt, memory::format_tag::ab);  // ab or ba
//             auto bias_md    = memory::desc({bias_tz}, bias_dt, memory::format_tag::ab);
//             auto dst_md     = memory::desc({dst_tz}, dst_dt, memory::format_tag::ab);

//             auto desc = matmul::desc(src_md, weights_md, bias_md, dst_md);

             
//             // auto *prim_desc = new matmul::primitive_desc(desc, eng);
//             // auto *prim = new matmul(*prim_desc);
//             auto prim_desc_ptr = std::make_shared<matmul::primitive_desc>(desc, eng);
//             auto prim_ptr      = std::make_shared<matmul>(*prim_desc_ptr);

//             g_prim[prim_key] = prim_ptr;
//             g_mm_prim_desc[prim_key] = prim_desc_ptr;
//             // g_prim.insert(std::pair<std::string, primitive *>(prim_key, prim));
//             // g_mm_prim_desc.insert(std::pair<std::string, matmul::primitive_desc *>(prim_key, prim_desc));
//             std::cout << "MatMul: save prim_key = " << prim_key << ", prim number = " << g_prim.size() << std::endl;
//         }

//         auto user_src_md     = memory::desc(src_tz, src_dt, memory::format_tag::ab);
//         auto user_weights_md = memory::desc(weights_tz, weights_dt, memory::format_tag::ab);  // ab or ba
//         auto user_bias_md    = memory::desc(bias_tz, bias_dt, memory::format_tag::ab);

//         auto user_src_memory_ptr     = std::make_shared<dnnl::memory>(user_src_md, eng, input);
//         auto user_weights_memory_ptr = std::make_shared<dnnl::memory>(user_weights_md, eng, weight);
//         auto user_bias_memory_ptr    = std::make_shared<dnnl::memory>(user_bias_md, eng, bias);

//         auto prim_desc_ptr = g_mm_prim_desc[prim_key];

//         auto src_memory_ptr     = user_src_memory_ptr;
//         auto weights_memory_ptr = user_weights_memory_ptr;
//         auto bias_memory_ptr    = user_bias_memory_ptr;
//         auto dst_memory_ptr     = std::make_shared<memory>(prim_desc_ptr->dst_desc(), eng, output);

//         if (prim_desc_ptr->src_desc() != user_src_memory_ptr->get_desc()) {
//             src_memory_ptr       = std::make_shared<memory>(prim_desc_ptr->src_desc(), eng);
//             auto reorder_src = reorder(*user_src_memory_ptr, *src_memory_ptr);
//             reorder_src.execute(stm, {{DNNL_ARG_FROM, *user_src_memory_ptr}, {DNNL_ARG_TO, *src_memory_ptr}});
//             stm.wait();
//         }

//         auto weights_mem_key = prim_key + "-weights";
//         auto weights_mem = g_memory.find(weights_mem_key);
//         if (weights_mem == g_memory.end()) {
//             if (prim_desc_ptr->weights_desc() != user_weights_memory_ptr->get_desc()) {
//                 weights_memory_ptr = std::make_shared<memory> (prim_desc_ptr->weights_desc(), eng);
//                 auto reorder_weights = reorder(*user_weights_memory_ptr, *weights_memory_ptr);
//                 reorder_weights.execute(stm, {{DNNL_ARG_FROM, *user_weights_memory_ptr}, {DNNL_ARG_TO, *weights_memory_ptr}});
//                 stm.wait();
//             }
//             g_memory[weights_mem_key] =   weights_memory_ptr; 
//         } else {
//             weights_memory_ptr = g_memory[weights_mem_key];
//         }

//         // if (prim_desc.weights_desc() != user_weights_memory.get_desc()) {
//         //     std::string prim_weights_key = prim_key + "-weights";
//         //     auto it_memory_created       = g_memory.find(prim_weights_key);
//         //     if (it_memory_created == g_memory.end()) {
//         //         std::cout << "MatMul: reorder user_weights_memory !!!" << std::endl;
//         //         weights_memory       = new memory(prim_desc.weights_desc(), eng);
//         //         auto reorder_weights = reorder(user_weights_memory, *weights_memory);
//         //         reorder_weights.execute(stm, {{DNNL_ARG_FROM, user_weights_memory}, {DNNL_ARG_TO, *weights_memory}});
//         //         g_memory.insert(std::pair<std::string, memory *>(prim_weights_key, weights_memory));
//         //     } else {
//         //         weights_memory = it_memory_created->second;
//         //     }
//         // }

        
//         auto prim_bias_key = prim_key + "-Bias";
//         auto bias_mem = g_memory.find(prim_bias_key);
//         if (bias_mem != g_memory.end()) {
//             if (prim_desc_ptr->bias_desc() != bias_memory_ptr->get_desc()) {
//                 bias_memory_ptr = std::make_shared<memory> (prim_desc_ptr->bias_desc(), eng);   
//                 auto reorder_bias = reorder(*user_bias_memory_ptr,*bias_memory_ptr);
//                 reorder_bias.execute(stm, {{DNNL_ARG_FROM, *user_bias_memory_ptr}, {DNNL_ARG_TO, *bias_memory_ptr}});
//                 stm.wait();
//                 g_memory[prim_bias_key] = bias_memory_ptr;
//             }
//         } else {
//             bias_memory_ptr = g_memory[prim_bias_key];
//         }




//         // if (prim_desc.bias_desc() != user_bias_memory.get_desc()) {
//         //     std::string prim_bias_key = prim_key + "-bias";
//         //     auto it_memory_created    = g_memory.find(prim_bias_key);
//         //     if (it_memory_created == g_memory.end()) {
//         //         std::cout << "MatMul: reorder user_bias_memory !!!" << std::endl;
//         //         bias_memory       = new memory(prim_desc.bias_desc(), eng);
//         //         auto reorder_bias = reorder(user_bias_memory, *bias_memory);
//         //         reorder_bias.execute(stm, {{DNNL_ARG_FROM, user_bias_memory}, {DNNL_ARG_TO, *bias_memory}});
//         //         g_memory.insert(std::pair<std::string, memory *>(prim_bias_key, bias_memory));
//         //     } else {
//         //         bias_memory = it_memory_created->second;
//         //     }
//         // }

//         it_prim_created = g_prim.find(prim_key);
//         it_prim_created->second->execute(stm, {{DNNL_ARG_SRC, *src_memory_ptr},
//                                                {DNNL_ARG_WEIGHTS, *weights_memory_ptr},
//                                                {DNNL_ARG_BIAS, *bias_memory_ptr},
//                                                {DNNL_ARG_DST, *dst_memory_ptr}});
//         stm.wait();
//         return true;
//     }

//     static void clean() {
//         g_memory.clear();
//         g_prim.clear();
//         g_ip_prim_desc.clear();
//         g_mm_prim_desc.clear();
//     }

// private:
//     /* data */
//     typedef std::unordered_map<std::string, std::shared_ptr<memory>> map_mem_t;
//     typedef std::unordered_map<std::string, std::shared_ptr<inner_product_forward::primitive_desc>> map_ip_primd_t;
//     typedef std::unordered_map<std::string, std::shared_ptr<matmul::primitive_desc>> map_mm_primd_t;
//     typedef std::unordered_map<std::string, std::shared_ptr<primitive>> map_prim_t;

//     static map_mem_t g_memory;
//     static map_ip_primd_t g_ip_prim_desc;
//     static map_mm_primd_t g_mm_prim_desc;
//     static map_prim_t g_prim;
// };


// typedef std::unordered_map<std::string, memory*> map_mem_t;
// typedef std::unordered_map<std::string, inner_product_forward::primitive_desc*> map_ip_primd_t;
// typedef std::unordered_map<std::string, matmul::primitive_desc*> map_mm_primd_t;
// typedef std::unordered_map<std::string, primitive*> map_prim_t;

// static map_mem_t g_memory;
// static map_ip_primd_t g_ip_prim_desc;
// static map_mm_primd_t g_mm_prim_desc;
// static map_prim_t g_prim;

#endif
