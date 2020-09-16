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
    BaseLayer *shape_calculator = CreateLayer(type_);
    std::vector<Blob *> input_blobs;
    BlobDesc blob_desc;
    for (auto &input_op : input_ops_) {
        blob_desc.dims.clear();
        for (auto dim : input_op->GetDims())
            blob_desc.dims.push_back((int)dim);
        Blob *blob = new Blob(blob_desc);
        input_blobs.push_back(blob);
    }
    std::vector<Blob *> output_blobs;
    for (int i = 0; i < outputs_name_.size(); i++) {
        Blob *blob = new Blob(blob_desc);
        output_blobs.push_back(blob);
    }
    Status ret = shape_calculator->InferShapeAhead(input_blobs, output_blobs, param_, resource_);
    if (ret == TNN_OK) {
        for (int i = 0; i < outputs_name_.size(); i++) {
            output_shapes.push_back(output_blobs[i]->GetBlobDesc().dims);
        }
    }

    for (auto &blob : input_blobs) {
        delete (blob);
    }
    for (auto &blob : output_blobs) {
        delete (blob);
    }
    input_blobs.clear();
    output_blobs.clear();
    delete (shape_calculator);
    return ret;
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
