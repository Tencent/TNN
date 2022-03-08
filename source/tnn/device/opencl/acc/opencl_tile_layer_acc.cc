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

#include "tnn/device/opencl/acc/opencl_unary_layer_acc.h"
namespace TNN_NS {

class OpenCLTileLayerAcc : public OpenCLLayerAcc {
public:
    virtual Status Init(Context *context, LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                        const std::vector<Blob *> &outputs) override;

    virtual ~OpenCLTileLayerAcc() override;

    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;
};

Status OpenCLTileLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init Tile Acc\n");
    Status ret = OpenCLLayerAcc::Init(context, param, resource, inputs, outputs);
    CHECK_TNN_OK(ret)

    auto output_dims = outputs[0]->GetBlobDesc().dims;
    auto input_dims  = inputs[0]->GetBlobDesc().dims;

    if ((input_dims.size() != 4 || output_dims.size() != 4) && (input_dims.size() != 6 || output_dims.size() != 6)) {
        LOGE("Tile Layer (OpenCL) only support 4-dim by now\n");
        return Status(TNNERR_INVALID_INPUT, "Tile Layer (OpenCL) only support 4-dim and 6-dim by now\n");
    }

    run_3d_ndrange_ = false;
    op_name_        = "Tile";
    std::string kernel_name;
    do {
        if (input_dims.size() == 6 && output_dims.size() == 6) {
            kernel_name = "Tile6D";
            break;
        }
        if (input_dims[1] == output_dims[1]) {
            kernel_name = "Tile_nhw";
            break;
        }
        kernel_name = "Tile";
    } while (0);

    // create kernel
    ret = CreateExecuteUnit(execute_units_[0], "tile", kernel_name, build_options_);
    if (ret != TNN_OK) {
        LOGE("create execute unit failed!\n");
        return ret;
    }

    return TNN_OK;
}

OpenCLTileLayerAcc::~OpenCLTileLayerAcc() {}

Status OpenCLTileLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    Status ret = OpenCLLayerAcc::Reshape(inputs, outputs);
    CHECK_TNN_OK(ret)

    auto output_dims = outputs[0]->GetBlobDesc().dims;
    auto input_dims  = inputs[0]->GetBlobDesc().dims;
    uint32_t idx     = SetExecuteUnit2DSizeInfoDefault(execute_units_[0], output_dims);
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)inputs[0]->GetHandle().base));
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)outputs[0]->GetHandle().base));

    if (input_dims.size() > 4 || output_dims.size() > 4) {
        execute_units_[0].ocl_kernel.setArg(idx++, input_dims.size() * sizeof(int), input_dims.data());
        execute_units_[0].ocl_kernel.setArg(idx++, output_dims.size() * sizeof(int), output_dims.data());

        return TNN_OK;
    }

    execute_units_[0].ocl_kernel.setArg(idx++, input_dims[0]);
    execute_units_[0].ocl_kernel.setArg(idx++, input_dims[1]);
    execute_units_[0].ocl_kernel.setArg(idx++, input_dims[2]);
    execute_units_[0].ocl_kernel.setArg(idx++, input_dims[3]);

    execute_units_[0].ocl_kernel.setArg(idx++, output_dims[0]);
    execute_units_[0].ocl_kernel.setArg(idx++, output_dims[1]);
    execute_units_[0].ocl_kernel.setArg(idx++, output_dims[2]);
    execute_units_[0].ocl_kernel.setArg(idx++, output_dims[3]);
    if (input_dims[1] != output_dims[1]) {
        execute_units_[0].ocl_kernel.setArg(idx++, output_dims[1] * output_dims[2] * output_dims[3]);
        execute_units_[0].ocl_kernel.setArg(idx++, output_dims[2] * output_dims[3]);
    }
    return TNN_OK;
}

REGISTER_OPENCL_ACC(Tile, LAYER_REPEAT)
REGISTER_OPENCL_LAYOUT(LAYER_REPEAT, DATA_FORMAT_NHC4W4);
}  // namespace TNN_NS
