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

#include "rknpu_base_layer.h"

#include <mutex>

#include "tnn/utils/npu_common_utils.h"

namespace TNN_NS {

RknpuBaseLayer::RknpuBaseLayer(LayerType type) {
    this->type_ = type;
}

RknpuBaseLayer::~RknpuBaseLayer(){};

Status RknpuBaseLayer::Init(Context *context, LayerParam *param, LayerResource *resource,
                            std::vector<std::shared_ptr<rk::nn::Tensor>> input_ops, rk::nn::Graph *graph,
                            std::vector<std::string> outputs) {
    param_        = param;
    resource_     = resource;
    input_ops_    = input_ops;
    outputs_name_ = outputs;
    graph_        = graph;

    Status ret = Convert();
    return ret;
}

void RknpuBaseLayer::SetLayerName(std::string layer_name) {
    layer_name_ = layer_name;
}

std::string RknpuBaseLayer::GetLayerName() {
    return layer_name_;
}

Status RknpuBaseLayer::GetOutputShape(int i, std::vector<int> &output_shape) {
    std::vector<std::vector<int>> output_shapes;
    Status ret = CalculateOutputShape(output_shapes);
    if (ret != TNN_OK)
        return ret;
    output_shape = output_shapes[i];
    return TNN_OK;
}

Status RknpuBaseLayer::CalculateOutputShape(std::vector<std::vector<int>> &output_shapes) {
    std::vector<BlobDesc> blob_descs;
    std::vector<Blob *> input_blobs;
    std::vector<Blob *> output_blobs;

    blob_descs.clear();
    for (auto &input_op : input_ops_) {
        BlobDesc blob_desc;
        for (auto dim : input_op->GetDims())
            blob_desc.dims.push_back((int)dim);
        blob_descs.emplace_back(blob_desc);
    }
    RETURN_ON_NEQ(NpuCommonUtils::CreateBlobs(blob_descs, input_blobs), TNN_OK);

    blob_descs.clear();
    for (int i = 0; i < outputs_name_.size(); i++) {
        BlobDesc blob_desc;
        blob_descs.emplace_back(blob_desc);
    }
    RETURN_ON_NEQ(NpuCommonUtils::CreateBlobs(blob_descs, output_blobs), TNN_OK);

    RETURN_ON_NEQ(NpuCommonUtils::CalculateOutputShape(type_, input_blobs, output_blobs, param_, resource_,
                                                       outputs_name_, output_shapes),
                  TNN_OK);

    RETURN_ON_NEQ(NpuCommonUtils::ReleaseBlobs(input_blobs, output_blobs), TNN_OK);

    return TNN_OK;
}

std::vector<std::shared_ptr<rk::nn::Tensor>> &RknpuBaseLayer::GetOutputOps() {
    return output_ops_;
}

std::map<LayerType, std::shared_ptr<RknpuLayerCreator>> &GetGlobalRknpuLayerCreatorMap() {
    static std::once_flag once;
    static std::shared_ptr<std::map<LayerType, std::shared_ptr<RknpuLayerCreator>>> creators;
    std::call_once(once, []() { creators.reset(new std::map<LayerType, std::shared_ptr<RknpuLayerCreator>>); });
    return *creators;
}

RknpuBaseLayer *CreateRknpuBaseLayer(LayerType type) {
    RknpuBaseLayer *cur_layer = nullptr;
    auto &layer_creater_map   = GetGlobalRknpuLayerCreatorMap();
    if (layer_creater_map.count(type) > 0) {
        cur_layer = layer_creater_map[type]->CreateRknpuBaseLayer();
    }
    return cur_layer;
}

}  // namespace TNN_NS
