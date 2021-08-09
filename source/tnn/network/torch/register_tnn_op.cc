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

#include "tnn/core/blob.h"
#include "tnn/network/torch/jit_util.h"
#include "tnn/network/torch/torch_convert.h"
#include "tnn/network/torch/torch_tnn_runtime.h"
#include "tnn/network/torch/torch_utils.h"
#include "torch/csrc/jit/runtime/custom_operator.h"
// #include "tnn/interpreter/tnn/model_packer.h"
#include <cuda_runtime.h>

#include "c10/cuda/CUDAStream.h"
#include "tnn/utils/blob_dump_utils.h"

namespace TNN_NS {
namespace runtime {

std::vector<at::Tensor> execute_engine(std::vector<at::Tensor> inputs,
                                        c10::intrusive_ptr<TNNEngine> compiled_engine) {
    auto scalar_type = inputs[0].scalar_type();
    auto input_names  = compiled_engine->input_names;
    auto output_names = compiled_engine->output_names;
    InputShapesMap inputs_shape_map;
    int input_idx = 0;
    for (auto &input : inputs) {
        inputs_shape_map[input_names[input_idx++]] = util::toDims(input.sizes());
    }

    if (!compiled_engine->is_init_) {
        auto interpreter = dynamic_cast<DefaultModelInterpreter *>(compiled_engine->ctx_->get_interpreter().get());
        interpreter->GetNetStructure()->inputs_shape_map = inputs_shape_map;
        compiled_engine->instance_->Init(compiled_engine->ctx_->get_interpreter(), inputs_shape_map);
        compiled_engine->is_init_ = true;

        if (inputs[0].is_cuda()) {
            c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream(inputs[0].device().index());
            compiled_engine->instance_->SetCommandQueue(stream.stream());
        }

        // ModelPacker package(interpreter->GetNetStructure(), interpreter->GetNetResource());
        // package.Pack("torch.tnnproto", "torch.tnnmodel");
    } else {
        auto interpreter = dynamic_cast<DefaultModelInterpreter *>(compiled_engine->ctx_->get_interpreter().get());
        for (auto input : inputs_shape_map) {
            if (input.second != interpreter->GetNetStructure()->inputs_shape_map[input.first]) {
                compiled_engine->instance_->Init(compiled_engine->ctx_->get_interpreter(), inputs_shape_map);
                break;
            }
        }
    }

    BlobMap input_blobs;
    BlobMap output_blobs;
    compiled_engine->instance_->GetAllInputBlobs(input_blobs);
    compiled_engine->instance_->GetAllOutputBlobs(output_blobs);

    for (int i = 0; i < input_names.size(); i++) {
        // cat not change trt inner data ptr by set blob handle, set input mat instead
        DeviceType device_type;
        BlobDesc blob_desc;
        ConvertToDeviceType(device_type, inputs[i].device());
        GetBlobDescFromTensor(blob_desc, inputs[i]);
        if (scalar_type == at::ScalarType::Half) {
            auto new_tensor = inputs[i].to(at::ScalarType::Float);
            auto input_mat = std::make_shared<Mat>(device_type, NCHW_FLOAT, blob_desc.dims, new_tensor.data_ptr());
            compiled_engine->instance_->SetInputMat(input_mat, MatConvertParam(), input_names[i]);
        } else {
            auto input_mat = std::make_shared<Mat>(device_type, NCHW_FLOAT, blob_desc.dims, inputs[i].data_ptr());
            compiled_engine->instance_->SetInputMat(input_mat, MatConvertParam(), input_names[i]);
        }
    }

    // DumpDeviceBlob(input_blobs[input_names[0]], cmd_queue, "tnn-input");

    compiled_engine->instance_->Forward();

    std::vector<at::Tensor> outputs(output_names.size());
    for (int i = 0; i < output_names.size(); i++) {
        std::shared_ptr<at::Tensor> tensor_ptr;
        CreateTensorByBlob(tensor_ptr, output_blobs[output_names[i]]);
        if (scalar_type == at::ScalarType::Half) {
            outputs[i] = std::move(tensor_ptr->to(at::ScalarType::Half));
        } else {
            outputs[i] = std::move(*tensor_ptr);
        }
    }
    // DumpDeviceBlob(output_blobs[output_names[0]], cmd_queue, "tnn-output");

    return outputs;
}

static auto TNNEngineTSRegistrtion = torch::class_<TNNEngine>("tnn", "Engine");

TORCH_LIBRARY(tnn, m) {
    // auto type_ptr = c10::detail::getTypePtr_<c10::intrusive_ptr<TNNEngine>>::call();
    m.def("execute_engine", execute_engine);
}

}  // namespace runtime
}  // namespace TNN_NS
