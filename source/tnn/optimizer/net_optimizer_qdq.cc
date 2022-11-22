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

#include "tnn/optimizer/net_optimizer_qdq.h"
#include "tnn/optimizer/QDQ/graph.h"

#include <map>
#include <memory>
#include <set>
#include <vector>

#include "tnn/core/common.h"
#include "tnn/core/layer_type.h"
#include "tnn/interpreter/layer_param.h"
#include "tnn/optimizer/net_optimizer_manager.h"
#include "tnn/optimizer/optimizer_const.h"

// #include <tnn/interpreter/tnn/model_packer.h>

namespace TNN_NS {

namespace optimizer {

    using namespace TNN_NS::QDQ;

    // P1 priority: should be fuse after bn scale fuse
    NetOptimizerRegister<NetOptimizerQDQ> g_net_optimizer_qdq(OptPriority::P0);

    std::string NetOptimizerQDQ::Strategy() {
        return kNetOptimizerQDQ;
    }

    bool NetOptimizerQDQ::IsSupported(const NetworkConfig &net_config) {
        if (net_config.device_type == DEVICE_CUDA) {
            return true;
        } else {
            return false;
        }
    }

    //! A --\                        A -> Q clone --\                                 
    //! B --+--> Concat --> Q   =>   B -> Q clone --+--> Concat
    //! C --/                        C -> Q clone --/

    // Status NetOptimizerQDQ::Optimize(NetStructure *structure, NetResource *resource) {
    //     Graph graph(structure, resource);
    //     std::unordered_map<std::string, std::string> rename_map;
    //     for (layer_id_t l = 0; l <= graph.GetMaxLayerId(); l++) {
    //         auto layer = graph.GetLayerById(l);
    //         if (layer->type == LAYER_CONCAT) {
    //             auto prodecessor_ids = graph.FindPredecessors(l);
    //             auto successor_ids = graph.FindSuccessors(l);

    //             assert(prodecessor_ids.size() > 0);
    //             if (successor_ids.size() == 0) continue;

    //             // match patten
    //             if (successor_ids.size() == 1) {
    //                 auto next_layer = graph.GetLayerById(successor_ids[0]);
    //                 if (next_layer->type != LAYER_QUANTIZE) continue;
    //             } else {
    //                 bool all_quantize = true;
    //                 for (auto nid : successor_ids) {
    //                     auto next_layer = graph.GetLayerById(nid);
    //                     all_quantize = all_quantize && (next_layer->type == LAYER_QUANTIZE);
    //                 }
    //                 if (!all_quantize) continue;
    //                 bool all_scale_equal = true;
    //                 for (auto nid : successor_ids) {
    //                     // Todo : check quantize scale
    //                 }
    //                 if (!all_scale_equal) continue;
    //             }

    //             // fuse patten
    //             auto q_layer_id = successor_ids[0];
    //             auto q_layer = graph.GetLayerById(q_layer_id);
    //             rename_map[q_layer->outputs[0]] = q_layer->inputs[0];
    //             std::vector<std::shared_ptr<LayerInfo>> q_clones;
    //             for (int i = 0; i < prodecessor_ids.size(); i++) {
    //                 auto prev_layer = graph.GetLayerById(prodecessor_ids[i]);
    //                 auto clone_layer = graph.CloneLayerById(q_layer_id, i);
    //                 // set clone input to prev output
    //                 clone_layer->inputs[0] = prev_layer->outputs[0];
    //                 // set concat input to clone output
    //                 layer->inputs[i] = clone_layer->outputs[0];
    //                 q_clones.push_back(clone_layer);
    //             }

    //             graph.EraseLayerById(q_layer_id);
    //             graph.InsertLayers(l, q_clones);
    //         } else {
    //             for (int i = 0; i < layer->inputs.size(); i++) {
    //                 if (rename_map.count(layer->inputs[i])) {
    //                     layer->inputs[i] = rename_map[layer->inputs[i]];
    //                 }
    //             }
    //         }
    //     }

    //     ModelPacker model_packer(structure, resource);
    //     model_packer.Pack("qdq.tnnproto", "qdq.tnnmodel"); 
    //     return TNN_OK;
    // }

    // dq -> HardSwish -> q  =>  QHardSwish
    Status NetOptimizerQDQ::Optimize(NetStructure *structure, NetResource *resource) {
        Graph graph(structure, resource);
        std::unordered_map<std::string, std::string> rename_map;

        // step 1
        //!   /-- Q - DQ - A                  /-- A
        //! X --- Q - DQ - B   =>  X - Q - DQ --- B  , if all the branchs have the same scale 
        //!   \-- Q - DQ - C                  \-- C
        // fuse Q Node
        for (layer_id_t l = 0; l <= graph.GetMaxLayerId(); l++) {
            auto layer         = graph.GetLayerById(l);
            auto successor_ids = graph.FindSuccessors(l);
            if (successor_ids.size() <= 1)
                continue;
            auto next_layer = graph.GetLayerById(successor_ids[0]);
            if (next_layer->type != LAYER_QUANTIZE)
                continue;

            // match pattern
            auto scale     = reinterpret_cast<QuantizeLayerResource *>(graph.GetLayerResByName(next_layer->name).get());
            auto ref_scale = scale->scale_handle.force_to<float *>()[0];
            bool quant_with_same_scale = true;
            for (int i = 1; i < successor_ids.size(); i++) {
                auto cur_layer = graph.GetLayerById(successor_ids[i]);
                if (cur_layer->type != LAYER_QUANTIZE) {
                    quant_with_same_scale = false;
                    break;
                } else {
                    auto scale =
                        reinterpret_cast<QuantizeLayerResource *>(graph.GetLayerResByName(next_layer->name).get());
                    auto compare_scale = scale->scale_handle.force_to<float *>()[0];
                    if (fabs(ref_scale - compare_scale) > 10e-6) {
                        quant_with_same_scale = false;
                        break;
                    }
                }
            }
            if (!quant_with_same_scale)
                continue;

            for (int i = 1; i < successor_ids.size(); i++) {
                auto cur_layer                    = graph.GetLayerById(successor_ids[i]);
                cur_layer->type                   = LAYER_NOT_SUPPORT;
                rename_map[cur_layer->outputs[0]] = next_layer->outputs[0];
            }
        }

        for (layer_id_t l = 0; l <= graph.GetMaxLayerId(); l++) {
            auto layer = graph.GetLayerById(l);
            for (int i = 0; i < layer->inputs.size(); i++) {
                if (rename_map.count(layer->inputs[i])) {
                    layer->inputs[i] = rename_map[layer->inputs[i]];
                }
            }
        }

        graph.EliminateDeadLayer();

        // fuse DQ Node
        for (layer_id_t l = 0; l <= graph.GetMaxLayerId(); l++) {
            auto layer         = graph.GetLayerById(l);
            auto successor_ids = graph.FindSuccessors(l);
            if (successor_ids.size() <= 1)
                continue;
            auto next_layer = graph.GetLayerById(successor_ids[0]);
            if (next_layer->type != LAYER_DEQUANTIZE)
                continue;

            // match pattern
            auto scale     = reinterpret_cast<QuantizeLayerResource *>(graph.GetLayerResByName(next_layer->name).get());
            auto ref_scale = scale->scale_handle.force_to<float *>()[0];
            bool quant_with_same_scale = true;
            for (int i = 1; i < successor_ids.size(); i++) {
                auto cur_layer = graph.GetLayerById(successor_ids[i]);
                if (cur_layer->type != LAYER_DEQUANTIZE) {
                    quant_with_same_scale = false;
                    break;
                } else {
                    auto scale =
                        reinterpret_cast<QuantizeLayerResource *>(graph.GetLayerResByName(next_layer->name).get());
                    auto compare_scale = scale->scale_handle.force_to<float *>()[0];
                    if (fabs(ref_scale - compare_scale) > 10e-6) {
                        quant_with_same_scale = false;
                        break;
                    }
                }
            }
            if (!quant_with_same_scale)
                continue;

            for (int i = 1; i < successor_ids.size(); i++) {
                auto cur_layer                    = graph.GetLayerById(successor_ids[i]);
                cur_layer->type                   = LAYER_NOT_SUPPORT;
                rename_map[cur_layer->outputs[0]] = next_layer->outputs[0];
            }
        }

        for (layer_id_t l = 0; l <= graph.GetMaxLayerId(); l++) {
            auto layer = graph.GetLayerById(l);
            for (int i = 0; i < layer->inputs.size(); i++) {
                if (rename_map.count(layer->inputs[i])) {
                    layer->inputs[i] = rename_map[layer->inputs[i]];
                }
            }
        }

        graph.EliminateDeadLayer();

        // step 2
        // dq -> HardSwish -> q  =>  QHardSwish
        for (layer_id_t l = 0; l <= graph.GetMaxLayerId(); l++) {
            auto layer = graph.GetLayerById(l);
            if (layer->type == LAYER_HARDSWISH) {
                auto prodecessor_ids = graph.FindPredecessors(l);
                auto successor_ids = graph.FindSuccessors(l);

                assert(prodecessor_ids.size() > 0);
                if (successor_ids.size() != 1) continue;
                if (prodecessor_ids.size() != 1) continue;

                // match patten
                auto prev_layer = graph.GetLayerById(prodecessor_ids[0]);
                if (prev_layer->type != LAYER_DEQUANTIZE) continue;
                auto next_layer = graph.GetLayerById(successor_ids[0]);
                if (next_layer->type != LAYER_QUANTIZE) continue;

                auto _prodecessor_ids = graph.FindPredecessors(prodecessor_ids[0]);
                auto __prodecessor_ids = graph.FindPredecessors(_prodecessor_ids[0]);
                auto check_layer = graph.GetLayerById(__prodecessor_ids[0]);
                // qhardswish after linear will cause trt error, linear has been changed from innerproduct to matmul+add
                if (check_layer->type == LAYER_ADD) continue;

                // Todo : check dq/q scale, should be equal

                // fuse patten
                rename_map[prev_layer->outputs[0]] = prev_layer->inputs[0];
                rename_map[next_layer->outputs[0]] = next_layer->inputs[0];

                layer->inputs[0] = rename_map[layer->inputs[0]];

                prev_layer->type = LAYER_NOT_SUPPORT;
                next_layer->type = LAYER_NOT_SUPPORT;

                layer->param->quantized = true;
                auto input_scale = reinterpret_cast<QuantizeLayerResource*>(graph.GetLayerResByName(prev_layer->name).get());
                auto output_scale = reinterpret_cast<QuantizeLayerResource*>(graph.GetLayerResByName(next_layer->name).get());
                auto input_scale_buf = std::make_shared<RawBuffer>(input_scale->scale_handle);
                auto output_scale_buf = std::make_shared<RawBuffer>(output_scale->scale_handle);
                graph.SetConstResByName(layer->inputs[0]+"_scale_data_", input_scale_buf);
                graph.SetConstResByName(layer->outputs[0]+"_scale_data_", output_scale_buf);

            } else {
                for (int i = 0; i < layer->inputs.size(); i++) {
                    if (rename_map.count(layer->inputs[i])) {
                        layer->inputs[i] = rename_map[layer->inputs[i]];
                    }
                }
            }
        }
        graph.EliminateDeadLayer();

        // ModelPacker model_packer(structure, resource);
        // model_packer.Pack("qdq.tnnproto", "qdq.tnnmodel"); 
        return TNN_OK;
    }

}  // namespace optimizer

}  // namespace TNN_NS
