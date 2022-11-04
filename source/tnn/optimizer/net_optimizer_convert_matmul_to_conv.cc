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

#include <map>
#include <memory>
#include <vector>

#include "tnn/core/layer_type.h"
#include "tnn/interpreter/layer_param.h"
#include "tnn/optimizer/graph_matcher/graph_matcher.h"
#include "tnn/optimizer/graph_matcher/graph_parser.h"
#include "tnn/optimizer/graph_matcher/ir.h"
#include "tnn/optimizer/graph_matcher/logger.h"
#include "tnn/optimizer/graph_matcher/text_graph_parser.h"
#include "tnn/optimizer/net_optimizer_manager.h"
#include "tnn/optimizer/net_optimizer_convert_matmul_to_conv.h"
#include "tnn/optimizer/optimizer_const.h"

namespace TNN_NS {

namespace optimizer {

    NetOptimizerRegister<NetOptimizerConvertMatMulToConv> g_net_optimizer_convert_matmul_to_conv(OptPriority::P1);

    std::string NetOptimizerConvertMatMulToConv::Strategy() {
        return kNetOptimizerConvertMatMulToConv;
    }

    bool NetOptimizerConvertMatMulToConv::IsSupported(const NetworkConfig &net_config) {
        if (net_config.device_type == DEVICE_ARM) {
            return true;
        }
        return false;
    }

    /*
     * On ARM, Conv1x1 performs better than MatMul, so MatMul is replaced by Conv1x1 here.
     * original graph,
     * graph(%in):
     *      %matmul_out = MatMul(%in)
     *      %out = Add(%matmul_out)
     *      return (%out)
     *
     * replaced graph,
     * graph(%in):
     *      %reshape0 = Reshape(%in)
     *      %permute0 = Permute(%reshape0)
     *      %conv = Convolution(%permute0)
     *      %permute1 = Permute(%conv)
     *
     *      %shape = Shape(%in)
     *      %slice = Slice(%shape)
     *      %new_shape = Concat(%slice)
     *
     *      %out = Reshape(%permute1, %new_shape)
     * */
    Status NetOptimizerConvertMatMulToConv::Optimize(NetStructure *structure, NetResource *resource) {
        if (!structure) {
            LOGE("Error: empty NetStructure\n");
            return Status(TNNERR_NET_ERR, "Error: empty NetStructure");
        }

        std::shared_ptr<Graph> graph = std::make_shared<Graph>();
        RETURN_IF_FAIL(graph->fromInterpreted(structure, resource));

        std::string graph_str = R"(
            graph(%in):
                %matmul_out = MatMul(%in)
                %out = Add(%matmul_out)
                return (%out)
        )";

        GraphRegistry registry;
        GraphParser graph_parser(&registry);
        std::shared_ptr<Graph> pattern = nullptr;
        if (graph_parser.parseFromString(graph_str)) {
            pattern = graph_parser.getGraph();
        } else {
            return Status(TNNERR_PARAM_ERR, "invalid pattern syntax.");
        }

        auto gen = [&](std::shared_ptr<AnchorGraph> in) -> std::shared_ptr<Graph> {
            if (in->inputs().size() != 1 || in->outputs().size() != 1) {
                return nullptr;
            }

            auto matmul_node = in->getNodeByTensorName(std::string("@matmul_out"));
            auto add_node    = in->getNodeByTensorName(std::string("@out"));
            if (!matmul_node || !add_node) {
                WARN("node of interest not found in convert matmul to conv optimizer");
                return nullptr;
            }

            const auto resource_map = resource->resource_map;
            const auto constant_map = resource->constant_map;

            std::shared_ptr<RawBuffer> matmul_buffer = nullptr;
            auto matmul_param = std::dynamic_pointer_cast<MatMulLayerParam>(matmul_node->info->param);
            if (matmul_param->weight_position == -1) {
                auto inputs = matmul_node->info->inputs;
                if (inputs.size() == 2 && constant_map.find(inputs[1]) != constant_map.end()) {
                    matmul_buffer = constant_map.at(inputs[1]);
                }
            } else if (matmul_param->weight_position == 1) {
                auto matmul_resource =
                    std::dynamic_pointer_cast<MatMulLayerResource>(resource_map.at(matmul_node->info->name));
                matmul_buffer = std::make_shared<RawBuffer>(matmul_resource->weight);
            }

            if (matmul_buffer == nullptr) {
                DEBUG("this matmul: %s can't be converted", matmul_node->name().c_str());
                return nullptr;
            }

            if (matmul_buffer->GetBufferDims().size() != 2) {
                DEBUG("this matmul: %s can't be converted", matmul_node->name().c_str());
                return nullptr;
            }

            std::shared_ptr<RawBuffer> add_buffer = nullptr;
            auto add_param = std::dynamic_pointer_cast<MultidirBroadcastLayerParam>(add_node->info->param);
            if (add_param->weight_input_index == -1) {
                auto inputs = add_node->info->inputs;
                if (inputs.size() == 2 && constant_map.find(inputs[1]) != constant_map.end()) {
                    add_buffer = constant_map.at(inputs[1]);
                }
            } else if (add_param->weight_input_index == 1) {
                auto add_resource =
                    std::dynamic_pointer_cast<EltwiseLayerResource>(resource_map.at(add_node->info->name));
                add_buffer = std::make_shared<RawBuffer>(add_resource->element_handle);
            }

            if (add_buffer == nullptr) {
                DEBUG("this add: %s can't be converted", add_node->name().c_str());
                return nullptr;
            }

            if (add_buffer->GetBufferDims().size() != 1 ||
                add_buffer->GetBufferDims().back() != matmul_buffer->GetBufferDims().back()) {
                DEBUG("this add: %s can't be converted", add_node->name().c_str());
                return nullptr;
            }

            INFO("found pattern at Node:%s", matmul_node->name().c_str());

            const std::string name_prefix = matmul_node->info->name + "_";

            // create Reshape
            auto g                             = std::make_shared<Graph>();
            const std::string in_name          = "input_1";
            const std::string reshape_in0_name = "reshape_in0";
            auto in1                           = g->getNodeOrCreatePlaceHolder(in_name);
            auto status                        = g->createNode(LAYER_RESHAPE, {in_name}, {reshape_in0_name});
            if (status != TNN_OK) {
                return nullptr;
            }
            auto new_reshape_in0_node         = g->getNodeByTensorName(reshape_in0_name);
            new_reshape_in0_node->info->param = std::make_shared<ReshapeLayerParam>();
            auto reshape_in0_param  = std::dynamic_pointer_cast<ReshapeLayerParam>(new_reshape_in0_node->info->param);
            auto matmul_buffer_dims = matmul_buffer->GetBufferDims();
            if (!reshape_in0_param) {
                return nullptr;
            }
            const int weight_dim0       = matmul_buffer_dims.front();
            const int weight_dim1       = matmul_buffer_dims.back();
            reshape_in0_param->shape    = {0, -1, weight_dim0, 1};
            reshape_in0_param->num_axes = reshape_in0_param->shape.size();

            // create Permute
            const std::string permute_in0_name = "permute_in0";
            status                             = g->createNode(LAYER_PERMUTE, {reshape_in0_name}, {permute_in0_name});
            if (status != TNN_OK) {
                return nullptr;
            }
            auto new_permute_in0_node         = g->getNodeByTensorName(permute_in0_name);
            new_permute_in0_node->info->param = std::make_shared<PermuteLayerParam>();
            auto permute_in0_param = std::dynamic_pointer_cast<PermuteLayerParam>(new_permute_in0_node->info->param);
            if (!permute_in0_param) {
                return nullptr;
            }
            permute_in0_param->orders = {0, 2, 1, 3};

            // create Convolution
            // generate conv weight
            matmul_buffer->Permute(weight_dim0, weight_dim1);
            matmul_buffer->SetBufferDims({weight_dim1, weight_dim0, 1, 1});

            const std::string conv_name = name_prefix + "conv";
            status                      = g->createNode(LAYER_CONVOLUTION, {permute_in0_name}, {conv_name});
            if (status != TNN_OK) {
                return nullptr;
            }
            auto new_conv_node         = g->getNodeByTensorName(conv_name);
            new_conv_node->info->param = std::make_shared<ConvLayerParam>();
            auto conv_param            = std::dynamic_pointer_cast<ConvLayerParam>(new_conv_node->info->param);
            if (!conv_param) {
                return nullptr;
            }
            conv_param->input_channel  = weight_dim0;
            conv_param->output_channel = weight_dim1;
            conv_param->kernels        = {1, 1};
            conv_param->pads           = {0, 0, 0, 0};
            conv_param->strides        = {1, 1};
            conv_param->dialations     = {1, 1};
            conv_param->bias           = 1;

            auto conv_resource                = std::make_shared<ConvLayerResource>();
            conv_resource->filter_handle      = *matmul_buffer;
            conv_resource->bias_handle        = *add_buffer;
            resource->resource_map[conv_name] = std::dynamic_pointer_cast<LayerResource>(conv_resource);

            // create Permute
            const std::string permute_out_name = "permute_out";
            status                             = g->createNode(LAYER_PERMUTE, {conv_name}, {permute_out_name});
            if (status != TNN_OK) {
                return nullptr;
            }
            auto new_permute_out_node         = g->getNodeByTensorName(permute_out_name);
            new_permute_out_node->info->param = std::make_shared<PermuteLayerParam>();
            auto permute_out_param = std::dynamic_pointer_cast<PermuteLayerParam>(new_permute_out_node->info->param);
            if (!permute_out_param) {
                return nullptr;
            }
            permute_out_param->orders = {0, 2, 1, 3};

            // create Shape
            const std::string shape_in0_name = "shape_in0";
            status                           = g->createNode(LAYER_SHAPE, {in_name}, {shape_in0_name});
            if (status != TNN_OK) {
                return nullptr;
            }
            auto new_shape_in0_node         = g->getNodeByTensorName(shape_in0_name);
            new_shape_in0_node->info->param = std::make_shared<LayerParam>();

            // create Slice
            const std::string slice_shape_name = "slice_shape";
            status = g->createNode(LAYER_STRIDED_SLICE_V2, {shape_in0_name}, {slice_shape_name});
            if (status != TNN_OK) {
                return nullptr;
            }
            auto new_slice_shape_node         = g->getNodeByTensorName(slice_shape_name);
            new_slice_shape_node->info->param = std::make_shared<StrideSliceV2LayerParam>();
            auto slice_shape_param =
                std::dynamic_pointer_cast<StrideSliceV2LayerParam>(new_slice_shape_node->info->param);
            if (!slice_shape_param) {
                return nullptr;
            }
            slice_shape_param->begins  = {0};
            slice_shape_param->ends    = {-1};
            slice_shape_param->strides = {1};
            slice_shape_param->axes    = {0};

            // create const layer for concat
            std::vector<int> concat_value = {weight_dim1};
            auto concat_constant_buffer   = std::make_shared<RawBuffer>(sizeof(int), (char *)(concat_value.data()));
            concat_constant_buffer->SetBufferDims({1});
            concat_constant_buffer->SetDataType(DATA_TYPE_INT32);
            const std::string concat_constant_name       = name_prefix + "constant_dim";
            resource->constant_map[concat_constant_name] = concat_constant_buffer;

            status = g->createNode(LAYER_CONST, {}, {concat_constant_name});
            if (status != TNN_OK) {
                return nullptr;
            }

            // create Concat
            const std::string concat_shape_name = "concat_shape";
            status = g->createNode(LAYER_CONCAT, {slice_shape_name, concat_constant_name}, {concat_shape_name});
            if (status != TNN_OK) {
                return nullptr;
            }
            auto new_concat_shape_node         = g->getNodeByTensorName(concat_shape_name);
            new_concat_shape_node->info->param = std::make_shared<ConcatLayerParam>();
            auto concat_shape_param = std::dynamic_pointer_cast<ConcatLayerParam>(new_concat_shape_node->info->param);
            if (!concat_shape_param) {
                return nullptr;
            }
            concat_shape_param->axis = 0;

            // create Reshape
            const std::string reshape_out_name = "reshape_out";
            status = g->createNode(LAYER_RESHAPE, {permute_out_name, concat_shape_name}, {reshape_out_name});
            if (status != TNN_OK) {
                return nullptr;
            }
            auto new_reshape_out_node         = g->getNodeByTensorName(reshape_out_name);
            new_reshape_out_node->info->param = std::make_shared<ReshapeLayerParam>();

            return g;
        };

        RETURN_IF_FAIL(graph->rewrite(pattern, gen));

        return TNN_OK;
    }

}  // namespace optimizer

}  // namespace TNN_NS
