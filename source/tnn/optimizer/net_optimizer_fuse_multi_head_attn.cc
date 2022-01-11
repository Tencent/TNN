// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "tnn/optimizer/net_optimizer_fuse_multi_head_attn.h"

#include <map>
#include <memory>
#include <vector>

#include "tnn/core/layer_type.h"
#include "tnn/interpreter/layer_param.h"
#include "tnn/optimizer/net_optimizer_manager.h"
#include "tnn/optimizer/optimizer_const.h"

namespace TNN_NS {

namespace optimizer {

    // P1 priority: should be fuse after bn scale fuse
    NetOptimizerRegister<NetOptimizerFuseMultiHeadAttn> g_net_optimizer_fuse_multi_head_attn(OptPriority::P1);

    std::string NetOptimizerFuseMultiHeadAttn::Strategy() {
        return kNetOptimizerFuseMultiHeadAttn;
    }

    bool NetOptimizerFuseMultiHeadAttn::IsSupported(const NetworkConfig &net_config) {
        auto device = net_config.device_type;
        auto precision = net_config.precision;
        if (device == DEVICE_CUDA && precision != PRECISION_HIGH) {
            //////////////////////////////////
            std::cout << "=== DEBUG, NetOptimizerFuse::IsSupported, device==DEVICE_CUDA, precision!=HIGH ===" << std::endl;
            //////////////////////////////////
            return true;
        }
        return false;
    }

    Status NetOptimizerFuseMultiHeadAttn::Optimize(NetStructure *structure, NetResource *resource) {
        if (!structure) {
            LOGE("Error: empty NetStructure\n");
            return Status(TNNERR_NET_ERR, "Error: empty NetStructure");
        }

        //////////////////////////////////
        std::cout << "=== DEBUG, NetOptimizerFuse::IsSupported, Optimize 0 ===" << std::endl;
        //////////////////////////////////
        std::vector<std::shared_ptr<LayerInfo>> layers_orig = structure->layers;
        if (layers_orig.size() <= 1) {
            return TNN_OK;
        }

        std::vector<std::shared_ptr<LayerInfo>> layers_fused = layers_orig;
        // std::shared_ptr does not have operator "=="
        // so std::remove_if is not supported, use a function instead.
        auto f_remove_mha_layer_info = [&layers_fused](std::shared_ptr<LayerInfo> layer_info) -> void {
            auto current_layer_info_iter = std::find(
                layers_fused.begin(), layers_fused.end(), layer_info);
            if (current_layer_info_iter==layers_fused.end()) {
                //std::cout << "=== remove if error ===" << std::endl;
                LOGE("NetOptimizerFuseMultiHeadAttention Error: invalid remove_if.\n");
                return;
            }
            layers_fused.erase(current_layer_info_iter);
        };

        std::multimap<std::string, std::shared_ptr<LayerInfo>> layer_i_multimap;
        std::map<std::string, std::shared_ptr<LayerInfo>> layer_o_map;
        for (int i=0; i<layers_orig.size(); i++) {
            for (const auto & in : layers_orig[i]->inputs) {
                layer_i_multimap.insert({in, layers_orig[i]});
            }
            for (const auto & out : layers_orig[i]->outputs) {
                auto ret = layer_o_map.insert({out, layers_orig[i]});
                if (!ret.second) {
                    LOGE("NetOptimizerFuseMultiHeadAttention Error: network has duplicate layer name.\n");
                    return TNN_OK;
                }
            }
        }

        auto f_get_layer_info_i_multimap = [&layer_i_multimap](std::string name) -> std::shared_ptr<LayerInfo> {
            auto layer_info_range = layer_i_multimap.equal_range(name);
            if (layer_info_range.first==layer_i_multimap.end() ||
                std::distance(layer_info_range.first, layer_info_range.second) != 1) {
                //std::cout << "=== DEBUG, distance = " << std::distance(layer_info_range.first, layer_info_range.second) << " ===" << std::endl;
                LOGE("NetOptimizerFuseMultiHeadAttention Error: invalid LayerInfo name.\n");
                return nullptr;
            }
            return layer_info_range.first->second;
        };
        
        auto f_get_layer_info_o_map = [&layer_o_map](std::string name) -> std::shared_ptr<LayerInfo> {
            auto layer_info_iter = layer_o_map.find(name);
            if (layer_info_iter==layer_o_map.end()) {
                //std::cout << "=== DEBUG, o_map find error ===" << std::endl; 
                LOGE("NetOptimizerFuseMultiHeadAttention Error: invalid LayerInfo name.\n");
                return nullptr;
            }
            return layer_info_iter->second;
        };
        //////////////////////////////////
        //std::cout << "=== DEBUG, NetOptimizerFuse::IsSupported, vec.size = " << layers_orig.size() << ", i_map.size = " << layer_i_multimap.size() << ", o_map.size = " << layer_o_map.size() << " ===" << std::endl;
        //////////////////////////////////
        


        // Layer check FUNCs 
        auto f_layer_is_matmul_possible_bmm = [](std::shared_ptr<LayerInfo> layer_info) -> bool {
            if (layer_info==nullptr) {
                return false;
            }
            if (layer_info->type!=LAYER_MATMUL ||
                layer_info->inputs.size()!=2) {
                return false;
            }
            auto matmul_param = dynamic_cast<MatMulLayerParam *>(layer_info->param.get());
            if (!matmul_param) {
                return false;
            }
            int matrix_a_num_dim = matmul_param->matrix_a_dims.size();
            int matrix_b_num_dim = matmul_param->matrix_b_dims.size();
            if (matrix_a_num_dim >0 && matrix_a_num_dim <3 &&
                matrix_a_num_dim != matrix_b_num_dim) {
                return false;
            }
            for (int dim=0; dim<matrix_a_num_dim-2; dim++) {
                if (matmul_param->matrix_a_dims[dim] != 
                    matmul_param->matrix_b_dims[dim]) {
                    return false;
                }
            }
            return true;
        };
        
        auto f_layer_is_permute_0213 = [](std::shared_ptr<LayerInfo> layer_info) -> bool {
            if (layer_info==nullptr) {
                return false;
            }
            if (layer_info->type!=LAYER_PERMUTE) {
                return false;
            }
            auto permute_param = dynamic_cast<PermuteLayerParam *>(layer_info->param.get());
            if (!permute_param) {
                return false;
            }
            if (permute_param->orders.size()==4 &&
                permute_param->orders[0]==0 && permute_param->orders[1]==2 &&
                permute_param->orders[2]==1 && permute_param->orders[3]==3) {
                return true;
            }
            return false;
        };

        auto f_layer_is_permute_0132 = [](std::shared_ptr<LayerInfo> layer_info) -> bool {
            if (layer_info==nullptr) {
                return false;
            }
            if (layer_info->type==LAYER_PERMUTE) {
                auto permute_param = dynamic_cast<PermuteLayerParam *>(layer_info->param.get());
                if (!permute_param) {
                    return false;
                }
                if (permute_param->orders.size()==4 &&
                    permute_param->orders[0]==0 && permute_param->orders[1]==1 &&
                    permute_param->orders[2]==3 && permute_param->orders[3]==2) {
                    return true;
                }
                return false;
            } else if (layer_info->type==LAYER_PERMUTEV2) {
                auto permutev2_param = dynamic_cast<PermuteV2LayerParam *>(layer_info->param.get());
                if (!permutev2_param) {
                    return false;
                }
                if ((permutev2_param->dim0==-1 && permutev2_param->dim1==-2) ||
                    (permutev2_param->dim0==-2 && permutev2_param->dim1==-1)) {
                    return true;
                }
                return false;
            }
            return false;
        };
        
        auto f_layer_is_mul_or_div_constant = [](std::shared_ptr<LayerInfo> layer_info) -> bool {
            if (layer_info==nullptr) {
                return false;
            }
            if (layer_info->type!=LAYER_MUL && layer_info->type!=LAYER_DIV) {
                return false;
            }
            auto layer_param = dynamic_cast<MultidirBroadcastLayerParam *>(layer_info->param.get());
            if (!layer_param ||
                layer_param->weight_input_index!=1) {
                return false;
            }
            return true;
        };

        auto f_layer_is_add_possible_mask = [](std::shared_ptr<LayerInfo> layer_info) -> bool {
            if (layer_info==nullptr) {
                return false;
            }
            if (layer_info->type!=LAYER_ADD) {
                return false;
            }
            auto layer_param = dynamic_cast<MultidirBroadcastLayerParam *>(layer_info->param.get());
            if (!layer_param ||
                layer_param->weight_input_index!=-1) {
                return false;
            }
            return true;
        };

        auto f_layer_is_softmax = [](std::shared_ptr<LayerInfo> layer_info) -> bool {
            if (layer_info==nullptr) {
                return false;
            }
            if (layer_info->type!=LAYER_SOFTMAX) {
                return false;
            }
            auto softmax_param = dynamic_cast<SoftmaxLayerParam *>(layer_info->param.get());
            if (!softmax_param ||
                (softmax_param->axis!=-1 && softmax_param->axis!=3)) {
                return false;
            }
            return true;
        };


        // The main func part. 
        for (int index; index < layers_orig.size(); index++) {
            auto layer_info_bmm_qk = layers_orig[index];
            if (f_layer_is_matmul_possible_bmm(layer_info_bmm_qk)) {
                std::string bmm_qk_input0_name = layer_info_bmm_qk->inputs[0];
                std::string bmm_qk_input1_name = layer_info_bmm_qk->inputs[1];
                std::string bmm_qk_output_name = layer_info_bmm_qk->outputs[0];
                auto layer_info_permute_q      = f_get_layer_info_o_map(bmm_qk_input0_name);
                auto layer_info_transpose_k    = f_get_layer_info_o_map(bmm_qk_input1_name);
                auto layer_info_mul_or_div     = f_get_layer_info_i_multimap(bmm_qk_output_name);
                if (!f_layer_is_permute_0213(layer_info_permute_q)) {
                    continue;
                } 
                if (!f_layer_is_permute_0132(layer_info_transpose_k)) {
                    continue;
                } 
                if (!f_layer_is_mul_or_div_constant(layer_info_mul_or_div)) {
                    continue;
                } 

                std::string transpose_input0_name = layer_info_transpose_k->inputs[0];
                std::string muldiv_output_name    = layer_info_mul_or_div->outputs[0];
                auto layer_info_permute_k         = f_get_layer_info_o_map(transpose_input0_name);
                auto layer_info_add_mask          = f_get_layer_info_i_multimap(muldiv_output_name);
                if (!f_layer_is_permute_0213(layer_info_permute_k)) {
                    continue;
                } 
                if (!f_layer_is_add_possible_mask(layer_info_add_mask)) {
                    continue;
                } 

                std::string add_mask_output_name = layer_info_add_mask->outputs[0];
                auto layer_info_softmax          = f_get_layer_info_i_multimap(add_mask_output_name);
                if (!f_layer_is_softmax(layer_info_softmax)) {
                    continue;
                }
                
                std::string softmax_output_name = layer_info_softmax->outputs[0];
                auto layer_info_bmm_v           = f_get_layer_info_i_multimap(softmax_output_name);
                if (!f_layer_is_matmul_possible_bmm(layer_info_bmm_v)) {
                    continue;
                }
                 
                std::string bmm_v_input1_name = layer_info_bmm_v->inputs[1];
                std::string bmm_v_output_name = layer_info_bmm_v->outputs[0];
                auto layer_info_permute_v     = f_get_layer_info_o_map(bmm_v_input1_name);
                auto layer_info_permute_o     = f_get_layer_info_i_multimap(bmm_v_output_name);
                if (!f_layer_is_permute_0213(layer_info_permute_v)) {
                    continue;
                } 
                if (!f_layer_is_permute_0213(layer_info_permute_o)) {
                    continue;
                } 

                // All checks passed. 
                // Multi-Head Attention Fuseable Block detected.
                // extract all Multi-Head Attention Related OPs from layers_fused,
                // except the last permute_out, replace it with a single new MultiHeadAttention Layer,
                std::cout << "=== DEBUG, layers_fused.size() 0 = " << layers_fused.size() << " ===" << std::endl;
                f_remove_mha_layer_info(layer_info_permute_q);
                f_remove_mha_layer_info(layer_info_permute_k);
                f_remove_mha_layer_info(layer_info_permute_v);
                f_remove_mha_layer_info(layer_info_transpose_k);
                f_remove_mha_layer_info(layer_info_bmm_qk);
                f_remove_mha_layer_info(layer_info_mul_or_div);
                f_remove_mha_layer_info(layer_info_add_mask);
                f_remove_mha_layer_info(layer_info_softmax);
                f_remove_mha_layer_info(layer_info_bmm_v);
                std::cout << "=== DEBUG, layers_fused.size() 1 = " << layers_fused.size() << " ===" << std::endl;

                /*
                layer_info_permute_o->type     = LAYER_MULTI_HEAD_ATTENTION;
                layer_info_permute_o->type_str = "MultiHeadAttention";
                layer_info_permute_o->inputs.clear();
                layer_info_permute_o->inputs.push_back(layer_info_permute_q->inputs[0]);
                layer_info_permute_o->inputs.push_back(layer_info_permute_k->inputs[0]);
                layer_info_permute_o->inputs.push_back(layer_info_permute_v->inputs[0]);
                layer_info_permute_o->param = std::make_shared<MultiHeadAttentionLayerParam>();
                */

                std::cout << "=== DEBUG, 80 idx = " << index << ", layer.name = " << layer_info_bmm_qk->name << std::endl;
                
            }
        }

        //structure->layers = layers_fused;

        return TNN_OK;
    }

}  // namespace optimizer

}  // namespace TNN_NS
