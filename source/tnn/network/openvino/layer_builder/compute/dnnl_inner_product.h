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

#ifndef __DNNL_INNER_PRODUCT__
#define __DNNL_INNER_PRODUCT__

#include "tnn/network/openvino/layer_builder/compute/dnnl_common.h"

// template <typename T_input, typename T_wei, typename T_bias, typename T_output>
// bool InnerProduct(engine eng, stream stm, T_input* input, T_wei* weight, T_bias* bias, T_output* output, int m, int n, int k)
// {
//     char type_input = (std::is_floating_point<T_input>::value) ? 'f' : 'b';
//     char type_weights = (std::is_floating_point<T_wei>::value) ? 'f' : 'b';
//     char type_bias = (std::is_floating_point<T_bias>::value) ? 'f' : 'b';
//     char type_output = (std::is_floating_point<T_output>::value) ? 'f' : 'b';

//     const void *address = static_cast<const void*>(weight);

//     std::stringstream weights_addr;
//     weights_addr << "InnerProduct-" << type_input << type_weights << type_bias << type_output \
//                  << '-' << m << '-' << n << '-' << k << '-' << address;
//     std::string prim_key = weights_addr.str();

//     memory::dims src_tz = { m, k };
//     memory::dims weights_tz = { n, k };
//     memory::dims bias_tz = { n };
//     memory::dims dst_tz = { m, n };

//     memory::data_type src_dt = (std::is_floating_point<T_input>::value) ? memory::data_type::f32 : memory::data_type::bf16;
//     memory::data_type weights_dt = (std::is_floating_point<T_wei>::value) ? memory::data_type::f32 : memory::data_type::bf16;
//     memory::data_type bias_dt = (std::is_floating_point<T_bias>::value) ? memory::data_type::f32 : memory::data_type::bf16;
//     memory::data_type dst_dt = (std::is_floating_point<T_output>::value) ? memory::data_type::f32 : memory::data_type::bf16;

//     auto it_prim_created = g_prim.find(prim_key);
//     if (it_prim_created == g_prim.end())
//     {
//         auto src_md     = memory::desc({ src_tz }, src_dt, memory::format_tag::any);
//         auto weights_md = memory::desc({ weights_tz }, weights_dt, memory::format_tag::any);
//         auto bias_md    = memory::desc({ bias_tz }, bias_dt, memory::format_tag::any);
//         auto dst_md     = memory::desc({ dst_tz }, dst_dt, memory::format_tag::any);
        
//         auto desc = inner_product_forward::desc(prop_kind::forward_inference, src_md, weights_md, bias_md, dst_md);

//         auto *prim_desc = new inner_product_forward::primitive_desc(desc, eng);

//         auto *prim = new inner_product_forward(*prim_desc);

//         g_prim.insert(std::pair<std::string, primitive *>(prim_key, prim));
//         g_ip_prim_desc.insert(std::pair<std::string, inner_product_forward::primitive_desc *>(prim_key, prim_desc));
//         std::cout << "InnerProduct: save prim_key = " << prim_key << ", prim number = " << g_prim.size() << std::endl;
//     }

//     auto user_src_md = memory::desc(src_tz, src_dt, memory::format_tag::nc);
//     auto user_weights_md = memory::desc(weights_tz, weights_dt, memory::format_tag::io); // oi or io
//     auto user_bias_md = memory::desc(bias_tz, bias_dt, memory::format_tag::x);

//     auto user_src_memory = memory(user_src_md, eng, input);
//     auto user_weights_memory = memory(user_weights_md, eng, weight);
//     auto user_bias_memory = memory(user_bias_md, eng, bias);

//     auto it_prim_desc_created = g_ip_prim_desc.find(prim_key);
//     if (it_prim_desc_created == g_ip_prim_desc.end()) {
//         std::cout << "InnerProduct error: can find g_ip_prim_desc = " << prim_key << std::endl;
//         return false;
//     }
//     inner_product_forward::primitive_desc prim_desc = *it_prim_desc_created->second;

//     auto src_memory = user_src_memory;
//     auto weights_memory = &user_weights_memory;
//     auto bias_memory = &user_bias_memory;
//     auto dst_memory = memory(prim_desc.dst_desc(), eng, output);

//     if (prim_desc.src_desc() != user_src_memory.get_desc()) {
//         static int index = 0;
//         index++;
//         if (index < 2)
//             std::cout << "InnerProduct: reorder user_src_memory !!!" << std::endl;

//         src_memory = memory(prim_desc.src_desc(), eng);
//         auto reorder_src = reorder(user_src_memory, src_memory);
//         reorder_src.execute(stm, { { DNNL_ARG_FROM, user_src_memory },
//                                           { DNNL_ARG_TO, src_memory } });
//     }

//     if (prim_desc.weights_desc() != user_weights_memory.get_desc()) {
//         std::string prim_weights_key = prim_key+"-weights";
//         auto it_memory_created = g_memory.find(prim_weights_key);
//         if (it_memory_created == g_memory.end()) {
//             std::cout << "InnerProduct: reorder user_weights_memory !!!" << std::endl;
//             weights_memory = new memory(prim_desc.weights_desc(), eng);
//             auto reorder_weights = reorder(user_weights_memory, *weights_memory);
//             reorder_weights.execute(stm, {
//                 { DNNL_ARG_FROM, user_weights_memory },
//                 { DNNL_ARG_TO, *weights_memory } });
//             g_memory.insert(std::pair<std::string, memory *>(prim_weights_key, weights_memory));
//         }
//         else {
//             weights_memory = it_memory_created->second;
//         }
//     }

//     if (prim_desc.bias_desc() != user_bias_memory.get_desc()) {
//         std::string prim_bias_key = prim_key+"-bias";
//         auto it_memory_created = g_memory.find(prim_bias_key);
//         if (it_memory_created == g_memory.end()) {
//             std::cout << "InnerProduct: reorder user_bias_memory !!!" << std::endl;
//             bias_memory = new memory(prim_desc.bias_desc(), eng);
//             auto reorder_bias = reorder(user_bias_memory, *bias_memory);
//             reorder_bias.execute(stm, {
//                 { DNNL_ARG_FROM, user_bias_memory },
//                 { DNNL_ARG_TO, *bias_memory } });
//             g_memory.insert(std::pair<std::string, memory *>(prim_bias_key, bias_memory));
//         }
//         else {
//             bias_memory = it_memory_created->second;
//         }
//     }

//     it_prim_created = g_prim.find(prim_key);
//     if (it_prim_created != g_prim.end()) {
//         it_prim_created->second->execute(stm, {
//             { DNNL_ARG_SRC, src_memory },
//             { DNNL_ARG_WEIGHTS, *weights_memory },
//             { DNNL_ARG_BIAS, *bias_memory },
//             { DNNL_ARG_DST, dst_memory } });
//     }
//     else {
//         std::cout << "InnerProduct: execute error, prim_key = " << prim_key << std::endl;
//         return false;
//     }
//     stm.wait();
//     return true;
// }

// // must be bf16 emgine
// template <typename T_input, typename T_wei, typename T_bias, typename T_output>
// bool InnerProduct_v2(engine eng, stream stm, T_input* input, T_wei* weight, T_bias* bias, T_output* output, int m, int n, int k)
// {
//     char type_input = (std::is_floating_point<T_input>::value) ? 'f' : 'b';
//     char type_weights = (std::is_floating_point<T_wei>::value) ? 'f' : 'b';
//     char type_bias = (std::is_floating_point<T_bias>::value) ? 'f' : 'b';
//     char type_output = (std::is_floating_point<T_output>::value) ? 'f' : 'b';

//     const void *address = static_cast<const void*>(weight);

//     std::stringstream weights_addr;
//     weights_addr << "InnerProduct2-" << type_input << type_weights << type_bias << type_output \
//                  << '-' << m << '-' << n << '-' << k << '-' << address;
//     std::string prim_key = weights_addr.str();

//     memory::dims src_tz = { m, k };
//     memory::dims weights_tz = { n, k };
//     memory::dims bias_tz = { n };
//     memory::dims dst_tz = { m, n };

//     memory::data_type src_dt = (std::is_floating_point<T_input>::value) ? memory::data_type::f32 : memory::data_type::bf16;
//     memory::data_type weights_dt = (std::is_floating_point<T_wei>::value) ? memory::data_type::f32 : memory::data_type::bf16;
//     memory::data_type bias_dt = (std::is_floating_point<T_bias>::value) ? memory::data_type::f32 : memory::data_type::bf16;
//     memory::data_type dst_dt = (std::is_floating_point<T_output>::value) ? memory::data_type::f32 : memory::data_type::bf16;

//     auto it_prim_created = g_prim.find(prim_key);
//     if (it_prim_created == g_prim.end())
//     {
//         auto src_md     = memory::desc({ src_tz }, memory::data_type::bf16, memory::format_tag::any);
//         auto weights_md = memory::desc({ weights_tz }, memory::data_type::bf16, memory::format_tag::any);
//         auto bias_md    = memory::desc({ bias_tz }, memory::data_type::bf16, memory::format_tag::any);
//         auto dst_md     = memory::desc({ dst_tz }, memory::data_type::bf16, memory::format_tag::any);
        
//         auto desc = inner_product_forward::desc(prop_kind::forward_inference, src_md, weights_md, bias_md, dst_md);

//         auto *prim_desc = new inner_product_forward::primitive_desc(desc, eng);

//         auto *prim = new inner_product_forward(*prim_desc);

//         g_prim.insert(std::pair<std::string, primitive *>(prim_key, prim));
//         g_ip_prim_desc.insert(std::pair<std::string, inner_product_forward::primitive_desc *>(prim_key, prim_desc));
//         std::cout << "InnerProduct: save prim_key = " << prim_key << ", prim number = " << g_prim.size() << std::endl;
//     }

//     auto user_src_md = memory::desc(src_tz, src_dt, memory::format_tag::nc);
//     auto user_weights_md = memory::desc(weights_tz, weights_dt, memory::format_tag::io); // oi or io
//     auto user_bias_md = memory::desc(bias_tz, bias_dt, memory::format_tag::x);

//     auto user_src_memory = memory(user_src_md, eng, input);
//     auto user_weights_memory = memory(user_weights_md, eng, weight);
//     auto user_bias_memory = memory(user_bias_md, eng, bias);

//     auto it_prim_desc_created = g_ip_prim_desc.find(prim_key);
//     if (it_prim_desc_created == g_ip_prim_desc.end()) {
//         std::cout << "InnerProduct error: can find g_ip_prim_desc = " << prim_key << std::endl;
//         return false;
//     }
//     inner_product_forward::primitive_desc prim_desc = *it_prim_desc_created->second;

//     auto src_memory = user_src_memory;
//     auto weights_memory = &user_weights_memory;
//     auto bias_memory = &user_bias_memory;
//     auto dst_memory = memory(prim_desc.dst_desc(), eng, output);

//     if (prim_desc.src_desc() != user_src_memory.get_desc()) {
//         static int index = 0;
//         index++;
//         if (index < 2)
//             std::cout << "InnerProduct: reorder user_src_memory !!!" << std::endl;

//         src_memory = memory(prim_desc.src_desc(), eng);
//         auto reorder_src = reorder(user_src_memory, src_memory);
//         reorder_src.execute(stm, { 
//             { DNNL_ARG_FROM, user_src_memory },
//             { DNNL_ARG_TO, src_memory } });
//     }

//     if (prim_desc.weights_desc() != user_weights_memory.get_desc()) {
//         std::string prim_weights_key = prim_key+"-weights";
//         auto it_memory_created = g_memory.find(prim_weights_key);
//         if (it_memory_created == g_memory.end()) {
//             std::cout << "InnerProduct: reorder user_weights_memory !!!" << std::endl;
//             weights_memory = new memory(prim_desc.weights_desc(), eng);
//             auto reorder_weights = reorder(user_weights_memory, *weights_memory);
//             reorder_weights.execute(stm, {
//                 { DNNL_ARG_FROM, user_weights_memory },
//                 { DNNL_ARG_TO, *weights_memory } });
//             g_memory.insert(std::pair<std::string, memory *>(prim_weights_key, weights_memory));
//         }
//         else {
//             weights_memory = it_memory_created->second;
//         }
//     }

//     if (prim_desc.bias_desc() != user_bias_memory.get_desc()) {
//         std::string prim_bias_key = prim_key+"-bias";
//         auto it_memory_created = g_memory.find(prim_bias_key);
//         if (it_memory_created == g_memory.end()) {
//             std::cout << "InnerProduct: reorder user_bias_memory !!!" << std::endl;
//             bias_memory = new memory(prim_desc.bias_desc(), eng);
//             auto reorder_bias = reorder(user_bias_memory, *bias_memory);
//             reorder_bias.execute(stm, {
//                 { DNNL_ARG_FROM, user_bias_memory },
//                 { DNNL_ARG_TO, *bias_memory } });
//             g_memory.insert(std::pair<std::string, memory *>(prim_bias_key, bias_memory));
//         }
//         else {
//             bias_memory = it_memory_created->second;
//         }
//     }

//     it_prim_created = g_prim.find(prim_key);
//     if (it_prim_created != g_prim.end()) {
//         it_prim_created->second->execute(stm, {
//             { DNNL_ARG_SRC, src_memory },
//             { DNNL_ARG_WEIGHTS, *weights_memory },
//             { DNNL_ARG_BIAS, *bias_memory },
//             { DNNL_ARG_DST, dst_memory } });
//     }
//     else {
//         std::cout << "InnerProduct: execute error, prim_key = " << prim_key << std::endl;
//         return false;
//     }
//     stm.wait();
//     return true;
// }

// template <typename T_input, typename T_wei, typename T_bias, typename T_output>
// bool InnerProduct_eltwise(engine eng, stream stm, T_input* input, T_wei* weight, T_bias* bias, T_output* output, int m, int n, int k)
// {
//     char type_input = (std::is_floating_point<T_input>::value) ? 'f' : 'b';
//     char type_weights = (std::is_floating_point<T_wei>::value) ? 'f' : 'b';
//     char type_bias = (std::is_floating_point<T_bias>::value) ? 'f' : 'b';
//     char type_output = (std::is_floating_point<T_output>::value) ? 'f' : 'b';

//     const void *address = static_cast<const void*>(weight);

//     std::stringstream weights_addr;
//     weights_addr << "InnerProductEltwise-" << type_input << type_weights << type_bias << type_output \
//                  << '-' << m << '-' << n << '-' << k << '-' << address;
//     std::string prim_key = weights_addr.str();

//     memory::dims src_tz = { m, k };
//     memory::dims weights_tz = { n, k };
//     memory::dims bias_tz = { n };
//     memory::dims dst_tz = { m, n };

//     memory::data_type src_dt = (std::is_floating_point<T_input>::value) ? memory::data_type::f32 : memory::data_type::bf16;
//     memory::data_type weights_dt = (std::is_floating_point<T_wei>::value) ? memory::data_type::f32 : memory::data_type::bf16;
//     memory::data_type bias_dt = (std::is_floating_point<T_bias>::value) ? memory::data_type::f32 : memory::data_type::bf16;
//     memory::data_type dst_dt = (std::is_floating_point<T_output>::value) ? memory::data_type::f32 : memory::data_type::bf16;

//     auto it_prim_created = g_prim.find(prim_key);
//     if (it_prim_created == g_prim.end())
//     {
//         auto src_md     = memory::desc({ src_tz }, memory::data_type::bf16, memory::format_tag::any);
//         auto weights_md = memory::desc({ weights_tz }, memory::data_type::bf16, memory::format_tag::any);
//         auto bias_md    = memory::desc({ bias_tz }, memory::data_type::bf16, memory::format_tag::any);
//         auto dst_md     = memory::desc({ dst_tz }, memory::data_type::bf16, memory::format_tag::any);
        
//         auto desc = inner_product_forward::desc(prop_kind::forward_inference, src_md, weights_md, bias_md, dst_md);

//         // Create primitive post-ops (ReLU)
//         const float scale = 1.0f;
//         const float alpha = 0.f;
//         const float beta = 0.f;
//         post_ops inner_product_ops;
//         inner_product_ops.append_eltwise(scale, algorithm::eltwise_relu, alpha, beta);
//         primitive_attr inner_product_attr;
//         inner_product_attr.set_post_ops(inner_product_ops);

//         auto *prim_desc = new inner_product_forward::primitive_desc(desc, inner_product_attr, eng);

//         auto *prim = new inner_product_forward(*prim_desc);

//         g_prim.insert(std::pair<std::string, primitive *>(prim_key, prim));
//         g_ip_prim_desc.insert(std::pair<std::string, inner_product_forward::primitive_desc *>(prim_key, prim_desc));
//         std::cout << "InnerProduct: save prim_key = " << prim_key << ", prim number = " << g_prim.size() << std::endl;
//     }

//     auto user_src_md = memory::desc(src_tz, src_dt, memory::format_tag::nc);
//     auto user_weights_md = memory::desc(weights_tz, weights_dt, memory::format_tag::io); // oi or io
//     auto user_bias_md = memory::desc(bias_tz, bias_dt, memory::format_tag::x);

//     auto user_src_memory = memory(user_src_md, eng, input);
//     auto user_weights_memory = memory(user_weights_md, eng, weight);
//     auto user_bias_memory = memory(user_bias_md, eng, bias);

//     auto it_prim_desc_created = g_ip_prim_desc.find(prim_key);
//     if (it_prim_desc_created == g_ip_prim_desc.end()) {
//         std::cout << "InnerProduct error: can find g_ip_prim_desc = " << prim_key << std::endl;
//         return false;
//     }
//     inner_product_forward::primitive_desc prim_desc = *it_prim_desc_created->second;

//     auto src_memory = user_src_memory;
//     auto weights_memory = &user_weights_memory;
//     auto bias_memory = &user_bias_memory;
//     auto dst_memory = memory(prim_desc.dst_desc(), eng, output);

//     if (prim_desc.src_desc() != user_src_memory.get_desc()) {
//         static int index = 0;
//         index++;
//         if (index < 2)
//             std::cout << "InnerProduct: reorder user_src_memory !!!" << std::endl;

//         src_memory = memory(prim_desc.src_desc(), eng);
//         auto reorder_src = reorder(user_src_memory, src_memory);
//         reorder_src.execute(stm, { 
//             { DNNL_ARG_FROM, user_src_memory },
//             { DNNL_ARG_TO, src_memory } });
//     }

//     if (prim_desc.weights_desc() != user_weights_memory.get_desc()) {
//         std::string prim_weights_key = prim_key+"-weights";
//         auto it_memory_created = g_memory.find(prim_weights_key);
//         if (it_memory_created == g_memory.end()) {
//             std::cout << "InnerProduct: reorder user_weights_memory !!!" << std::endl;
//             weights_memory = new memory(prim_desc.weights_desc(), eng);
//             auto reorder_weights = reorder(user_weights_memory, *weights_memory);
//             reorder_weights.execute(stm, {
//                 { DNNL_ARG_FROM, user_weights_memory },
//                 { DNNL_ARG_TO, *weights_memory } });
//             g_memory.insert(std::pair<std::string, memory *>(prim_weights_key, weights_memory));
//         }
//         else {
//             weights_memory = it_memory_created->second;
//         }
//     }

//     if (prim_desc.bias_desc() != user_bias_memory.get_desc()) {
//         std::string prim_bias_key = prim_key+"-bias";
//         auto it_memory_created = g_memory.find(prim_bias_key);
//         if (it_memory_created == g_memory.end()) {
//             std::cout << "InnerProduct: reorder user_bias_memory !!!" << std::endl;
//             bias_memory = new memory(prim_desc.bias_desc(), eng);
//             auto reorder_bias = reorder(user_bias_memory, *bias_memory);
//             reorder_bias.execute(stm, {
//                 { DNNL_ARG_FROM, user_bias_memory },
//                 { DNNL_ARG_TO, *bias_memory } });
//             g_memory.insert(std::pair<std::string, memory *>(prim_bias_key, bias_memory));
//         }
//         else {
//             bias_memory = it_memory_created->second;
//         }
//     }

//     it_prim_created = g_prim.find(prim_key);
//     if (it_prim_created != g_prim.end()) {
//         it_prim_created->second->execute(stm, {
//             { DNNL_ARG_SRC, src_memory },
//             { DNNL_ARG_WEIGHTS, *weights_memory },
//             { DNNL_ARG_BIAS, *bias_memory },
//             { DNNL_ARG_DST, dst_memory } });
//     }
//     else {
//         std::cout << "InnerProduct: execute error, prim_key = " << prim_key << std::endl;
//         return false;
//     }
//     stm.wait();
//     return true;
// }

#endif
