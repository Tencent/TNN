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
#include "tnn/interpreter/tnn/model_packer.h"
#include <cuda_runtime.h>

#include "c10/cuda/CUDAStream.h"
#include "tnn/utils/blob_dump_utils.h"
#include "tnn/interpreter/tnn/model_interpreter.h"

namespace TNN_NS {
namespace runtime {

std::vector<at::Tensor> execute_engine(std::vector<at::Tensor> inputs,
                                        c10::intrusive_ptr<TNNEngine> compiled_engine) {
    auto is_cuda = inputs[0].is_cuda();
    auto scalar_type = inputs[0].scalar_type();
    auto input_names  = compiled_engine->input_names;
    auto output_names = compiled_engine->output_names;
    InputShapesMap inputs_shape_map;
    InputDataTypeMap inputs_data_type_map;

    int input_idx = 0;
    for (auto &input : inputs) {
        inputs_shape_map[input_names[input_idx]] = util::toDims(input.sizes());
        BlobDesc blob_desc;
        GetBlobDescFromTensor(blob_desc, inputs[input_idx]);
        // binding input data type
        inputs_data_type_map[input_names[input_idx++]] = blob_desc.data_type;
    }

    if (!compiled_engine->is_init_) {
        auto interpreter = dynamic_cast<ModelInterpreter *>(compiled_engine->ctx_->get_interpreter().get());
        interpreter->GetNetStructure()->inputs_shape_map = inputs_shape_map;
        interpreter->GetNetStructure()->input_data_type_map = inputs_data_type_map;
        InputShapesMap min_shape;
        InputShapesMap max_shape;
        if (compiled_engine->max_inputs_shape.size()) {
            int input_idx = 0;
            for (auto &input : inputs) {
                min_shape[input_names[input_idx]] = compiled_engine->min_inputs_shape[input_idx];
                max_shape[input_names[input_idx]] = compiled_engine->max_inputs_shape[input_idx];
                input_idx++;
            }
        } else {
            min_shape = inputs_shape_map;
            max_shape = inputs_shape_map;
        }

        // ModelPacker package(interpreter->GetNetStructure(), interpreter->GetNetResource());
        // package.Pack("torch.tnnproto", "torch.tnnmodel");
        compiled_engine->instance_->Init(compiled_engine->ctx_->get_interpreter(), min_shape, max_shape);
        compiled_engine->is_init_ = true;
        compiled_engine->instance_->SetCpuNumThreads(10);
    }

    if (inputs[0].is_cuda()) {
        c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream(inputs[0].device().index());
        compiled_engine->instance_->SetCommandQueue(stream.stream());
    }

    if (is_cuda) {
        compiled_engine->instance_->Reshape(inputs_shape_map);
    }

    BlobMap input_blobs;
    BlobMap output_blobs;
    compiled_engine->instance_->GetAllInputBlobs(input_blobs);
    compiled_engine->instance_->GetAllOutputBlobs(output_blobs);

    // void *cmd_queue;
    // compiled_engine->instance_->GetCommandQueue(&cmd_queue);

    // tnn needs contingous torch tensor
    std::vector<at::Tensor> contig_inputs{};
    contig_inputs.reserve(inputs.size());

    for (int i = 0; i < input_names.size(); i++) {
        // set blob handle directly
        DeviceType device_type;
        BlobDesc blob_desc;
        ConvertToDeviceType(device_type, inputs[i].device());
        GetBlobDescFromTensor(blob_desc, inputs[i]);
        auto contig_input = inputs[i].contiguous();
        // extend the lifetime of contig tensors
        contig_inputs.emplace_back(contig_input);

        BlobHandle handle;
        handle.base = contig_input.data_ptr();
        if (is_cuda) {
            input_blobs[input_names[i]]->SetHandle(handle);
            input_blobs[input_names[i]]->SetBlobDesc(blob_desc);
        } else {
            auto blob_handle = input_blobs[input_names[i]]->GetHandle();
            // auto input_mat = std::make_shared<Mat>(DEVICE_NAIVE, NCHW_FLOAT, blob_desc.dims, handle.base);
            // compiled_engine->instance_->SetInputMat(input_mat, MatConvertParam(), input_names[i]);
            memcpy(blob_handle.base, handle.base, DimsVectorUtils::Count(blob_desc.dims) * sizeof(float));
        }
        // DumpDeviceBlob(input_blobs[input_names[i]], cmd_queue, "tnn-input-"+input_names[i]);
    }

    if (!is_cuda) {
        compiled_engine->instance_->Forward();
    }

    std::vector<at::Tensor> outputs(output_names.size());
    for (int i = 0; i < output_names.size(); i++) {
        // output blob data type is consistent with the input tensor, no need to convert tensor type
        auto desc = output_blobs[output_names[i]]->GetBlobDesc();
        c10::Device device(c10::kCPU);
        ConvertToTorchDevice(device, desc.device_type);
        at::ScalarType scalar_type;
        ConvertToTorchDataType(scalar_type, desc.data_type);
        outputs[i] = std::move(at::empty(ConvertDimsToIntArrayRef(desc.dims), {device.type()}).to(scalar_type).contiguous());

        BlobHandle handle;
        handle.base = outputs[i].data_ptr();;
        if (is_cuda) {
            output_blobs[output_names[i]]->SetHandle(handle);
        } else {
            auto blob_handle = output_blobs[output_names[i]]->GetHandle();
            memcpy(handle.base, blob_handle.base, DimsVectorUtils::Count(desc.dims) * sizeof(float));
        }

        // DumpDeviceBlob(output_blobs[output_names[i]], cmd_queue, "tnn-output-"+output_names[i]);
    }

    // use torch memory management
    if (is_cuda) {
        compiled_engine->instance_->Forward();
    }

    return outputs;
}

static auto TNNEngineTSRegistrtion = 
    torch::class_<TNNEngine>("tnn", "Engine")
        .def_pickle(
        [](const c10::intrusive_ptr<TNNEngine>& self) -> std::vector<std::string> {
            auto interpreter = dynamic_cast<ModelInterpreter *>(self->ctx_->get_interpreter().get());
            ModelPacker packer(interpreter->GetNetStructure(), interpreter->GetNetResource());
            std::string proto_s;
            std::string model_s;
            std::string input_names = Serialize(self->input_names);
            std::string output_names = Serialize(self->output_names);

            std::vector<std::string> input_shapes_vec;
            for (auto &shape : self->min_inputs_shape) {
                input_shapes_vec.emplace_back(Serialize(shape, TORCH_INT_DELIM));
            }
            std::string min_input_shapes = Serialize(input_shapes_vec);
            input_shapes_vec.clear();
            for (auto &shape : self->max_inputs_shape) {
                input_shapes_vec.emplace_back(Serialize(shape, TORCH_INT_DELIM));
            }
            std::string max_input_shapes = Serialize(input_shapes_vec);

            packer.GetSerialization(proto_s, model_s);

            std::vector<int> config_vec = {self->network_config_.device_type,
                                           self->network_config_.device_id,
                                           self->network_config_.precision,
                                           self->network_config_.share_memory_mode};
            std::string config_s = Serialize(config_vec);

            std::vector<std::string> contents;
            contents.emplace_back(proto_s);
            contents.emplace_back(model_s);
            contents.emplace_back(input_names);
            contents.emplace_back(output_names);
            contents.emplace_back(min_input_shapes);
            contents.emplace_back(max_input_shapes);
            contents.emplace_back(config_s);

            std::string cache_str;
            if (self->is_init_)
                dynamic_cast<ModelInterpreter *>(self->instance_->GetInterpreter().get())->GetCache(cache_str);
            contents.emplace_back(cache_str);

            return contents;
        },
        [](std::vector<std::string> seralized_engine) -> c10::intrusive_ptr<TNNEngine> {
            return c10::make_intrusive<TNNEngine>(seralized_engine);
        });

TORCH_LIBRARY(tnn, m) {
    // auto type_ptr = c10::detail::getTypePtr_<c10::intrusive_ptr<TNNEngine>>::call();
    m.def("execute_engine", execute_engine);
}

}  // namespace runtime
}  // namespace TNN_NS
