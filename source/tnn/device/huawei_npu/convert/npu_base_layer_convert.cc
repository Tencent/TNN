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

#include "npu_base_layer_convert.h"

namespace TNN_NS {

OperatorInfo::OperatorInfo() = default;

OperatorInfo::~OperatorInfo() {}

OperatorInfo::OperatorInfo(std::shared_ptr<ge::Operator> op) {
    this->op_ = op;
}
OperatorInfo::OperatorInfo(std::shared_ptr<ge::Operator> op, std::vector<int> shape) {
    this->op_    = op;
    this->shape_ = shape;
}

shared_ptr<ge::Operator> OperatorInfo::GetOperator() {
    return op_;
}

std::vector<int> OperatorInfo::GetShape() {
    return shape_;
}
void OperatorInfo::SetShape(std::vector<int> shape) {
    this->shape_ = shape;
}
void OperatorInfo::SetOperator(std::shared_ptr<ge::Operator> op) {
    this->op_ = op;
}

NpuBaseLayer::NpuBaseLayer(LayerType type) {
    this->type_ = type;
}

NpuBaseLayer::~NpuBaseLayer(){};

Status NpuBaseLayer::Init(Context *context, LayerParam *param, LayerResource *resource,
                          std::vector<std::shared_ptr<OperatorInfo>> input_ops, AbstractDevice *device,
                          std::vector<std::string> outputs) {
    param_        = param;
    resource_     = resource;
    input_ops_    = input_ops;
    outputs_name_ = outputs;
    // Convert all layers
    Status ret = Convert();
    return ret;
}

void NpuBaseLayer::SetLayerName(std::string layer_name) {
    layer_name_ = layer_name;
}

std::string NpuBaseLayer::GetLayerName() {
    return layer_name_;
}

Status NpuBaseLayer::SetOutputOps() {
    // calculate the output shape
    // all output index (output shape/ops) follow the outputs_name_ attribute
    std::vector<std::vector<int>> output_shapes;
    Status ret = NpuBaseLayer::CalculateOutputShape(output_shapes);
    if (ret != TNN_OK)
        return ret;
    for (int i = 0; i < outputs_name_.size(); i++) {
        output_ops_[i]->SetShape(output_shapes[i]);
    }
    return TNN_OK;
}

Status NpuBaseLayer::GetOutputShape(int i, std::vector<int> &output_shape) {
    std::vector<std::vector<int>> output_shapes;
    Status ret = CalculateOutputShape(output_shapes);
    if (ret != TNN_OK)
        return ret;
    output_shape = output_shapes[i];
    return TNN_OK;
}

Status NpuBaseLayer::CalculateOutputShape(std::vector<std::vector<int>> &output_shapes) {
    BaseLayer *shape_calculator = CreateLayer(type_);
    std::vector<Blob *> input_blobs;
    BlobDesc blob_desc;
    for (auto &input_op : input_ops_) {
        blob_desc.dims = input_op->GetShape();
        Blob *blob     = new Blob(blob_desc);
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

std::vector<std::shared_ptr<OperatorInfo>> &NpuBaseLayer::GetOutputOps() {
    return output_ops_;
}

std::map<LayerType, std::shared_ptr<NpuLayerCreator>> &GetGlobalNpuLayerCreatorMap() {
    static std::once_flag once;
    static std::shared_ptr<std::map<LayerType, std::shared_ptr<NpuLayerCreator>>> creators;
    std::call_once(once, []() { creators.reset(new std::map<LayerType, std::shared_ptr<NpuLayerCreator>>); });
    return *creators;
}

NpuBaseLayer *CreateNpuBaseLayer(LayerType type) {
    NpuBaseLayer *cur_layer = nullptr;
    auto &layer_creater_map = GetGlobalNpuLayerCreatorMap();
    if (layer_creater_map.count(type) > 0) {
        cur_layer = layer_creater_map[type]->CreateNpuBaseLayer();
    }
    return cur_layer;
}

}  // namespace TNN_NS
