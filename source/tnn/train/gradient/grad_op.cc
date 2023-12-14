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

#include "tnn/train/gradient/grad_op.h"

namespace TNN_NS {

GradOp::GradOp() {}

GradOp::~GradOp() {}

Status GradOp::RegisterGradOpCreator(DeviceType device, LayerType type, GradOpCreator grad_op_creator) {
    GetGradOpCreatorMap()[{device, type}] = grad_op_creator;
    return TNN_OK;
}

GradOpPtr GradOp::CreateGradOp(DeviceType device, LayerType type) {
    auto &grad_op_creator_map = GetGradOpCreatorMap();
    if (grad_op_creator_map.count({device, type}) > 0) {
        return grad_op_creator_map[{device, type}]();
    }
    return nullptr;
}

std::map<std::pair<DeviceType, LayerType>, GradOpCreator> &GradOp::GetGradOpCreatorMap() {
    static std::map<std::pair<DeviceType, LayerType>, GradOpCreator> grad_op_creator_map;
    return grad_op_creator_map;
}

// struct GradComputeConext {
//     std::vector<Blob *> fw_inputs;          
//     std::vector<Blob *> fw_outputs;
//     std::vector<RawBuffer *> train_resource;

//     std::vector<Blob *> input_grads;
//     std::vector<Blob *> output_grads;
//     std::vector<Blob *> train_resource_grads;

//     std::vector<DimsVector> fw_input_dims;
//     std::vector<DimsVector> fw_output_dims;
//     std::vector<DimsVector> train_resource_dims;
// };

}  // namespace TNN_NS
