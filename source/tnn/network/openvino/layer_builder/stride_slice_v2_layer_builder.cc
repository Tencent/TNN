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

#include <cmath>
#include <memory>

#include <ngraph/node.hpp>
#include <ngraph/ngraph.hpp>
#include <ngraph/op/op.hpp>
#include <ngraph/opsets/opset.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <inference_engine.hpp>

#include "tnn/layer/base_layer.h"
#include "tnn/network/openvino/layer_builder/openvino_layer_builder.h"
#include "tnn/extern_wrapper/foreign_blob.h"
#include "tnn/extern_wrapper/foreign_tensor.h"
#include "tnn/network/openvino/openvino_types.h"

#include "tnn/network/openvino/custom_layer/custom_stride_slice_v2.h"
#include "tnn/network/openvino/utils.h"

namespace TNN_NS {

DECLARE_OPENVINO_LAYER_BUILDER(StrideSliceV2, LAYER_STRIDED_SLICE_V2);

Status StrideSliceV2OVLayerBuilder::Build() {

    auto paramlist = dynamic_cast<StrideSliceV2LayerParam*>(param_);

     if (GetInputNodes().size() <=0) {
        LOGE("Error: 0 input nodes\n");
        return TNNERR_INIT_LAYER;
    }

    // // Todo : build error, zero dims is not allowed???
    // ngraph::Shape strideSliceShape;
    // auto dims = input_node[0]->get_output_shape(0);
    // auto dims_len = dims.size();
    // strideSliceShape.push_back(dims_len);

    // std::vector<int> begins(dims_len);
    // std::vector<int> ends(dims_len);
    // std::vector<int> strides(dims_len, 1);
    // std::vector<int64_t> begin_mask(dims_len, 1);
    // std::vector<int64_t> end_mask(dims_len, 1);
    // for (int i = 0; i < paramlist->axes.size(); i++) {
    //     if (paramlist->begins[i] < 0) {
    //         begins[paramlist->axes[i]] = paramlist->begins[i] + dims[i];
    //     } else {
    //         begins[paramlist->axes[i]] = paramlist->begins[i];
    //     }

    //     if (paramlist->ends[i] == INT_MAX) {
    //         ends[paramlist->axes[i]] = dims[i];
    //     } else if (paramlist->ends[i] < 0) {
    //         ends[paramlist->axes[i]] = paramlist->ends[i] + dims[i];
    //     } else {
    //         ends[paramlist->axes[i]] = paramlist->ends[i];
    //     }
    //     strides[paramlist->axes[i]] = paramlist->strides[i];
    //     begin_mask[paramlist->axes[i]] = 0; // 0 means valid at this dim
    //     end_mask[paramlist->axes[i]] = 0; // 0 means valid at this dim
    // }

    // auto beginNode = std::make_shared<ngraph::op::Constant>(
    //     ngraph::element::Type_t::i32, strideSliceShape, begins);
    // auto endNode = std::make_shared<ngraph::op::Constant>(
    //     ngraph::element::Type_t::i32, strideSliceShape, ends);
    // auto strideNode = std::make_shared<ngraph::op::Constant>(
    //     ngraph::element::Type_t::i32, strideSliceShape, strides);

    // auto strideSliceNode = std::make_shared<ngraph::op::v1::StridedSlice>(
    //     input_node[0]->output(0), beginNode, endNode, strideNode, begin_mask, end_mask);
    
    // strideSliceNode->validate_and_infer_types();
    // strideSliceNode->set_friendly_name(paramlist->name);

    // ngraph::NodeVector outputNodes;
    // outputNodes.push_back(strideSliceNode);
    // SetOutputTensors(outputNodes);

    // auto input_dims = GetInputBlobs()[0]->GetBlobDesc().dims;
    // auto output_dims = GetOutputBlobs()[0]->GetBlobDesc().dims;

    ADD_CUSTOM_NODE(StrideSliceV2, paramlist->name);

    return TNN_OK;
}

REGISTER_OPENVINO_LAYER_BUILDER(StrideSliceV2, LAYER_STRIDED_SLICE_V2);
REGISTER_CUSTOM_TYPE(LAYER_STRIDED_SLICE_V2);
}