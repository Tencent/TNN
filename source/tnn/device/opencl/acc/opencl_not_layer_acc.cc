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

#include "tnn/device/opencl/acc/opencl_layer_acc.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

class OpenCLNotLayerAcc : public OpenCLLayerAcc {
public:
    virtual Status Init(Context *context, LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                        const std::vector<Blob *> &outputs) override;

    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;

private:
    virtual std::vector<DataType> SupportDataType(int dims_size, BlobType blob_type) override;
};

Status OpenCLNotLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                 const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init Not Acc\n");
    Status ret = OpenCLLayerAcc::Init(context, param, resource, inputs, outputs);
    CHECK_TNN_OK(ret)

    execute_units_.resize(1);

    ret = CreateExecuteUnit(execute_units_[0], "not", "Not");
    if (ret != TNN_OK) {
        LOGE("create equal execute unit failed!\n");
        return ret;
    }

    return TNN_OK;
}

Status OpenCLNotLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Not Acc Reshape\n");
    Status ret = OpenCLLayerAcc::Reshape(inputs, outputs);
    CHECK_TNN_OK(ret)

    auto input  = inputs[0];
    auto output = outputs[0];

    auto output_dims = output->GetBlobDesc().dims;

    auto &unit = execute_units_[0];
    int idx    = SetExecuteUnit2DSizeInfoDefault(unit, output_dims);
    unit.ocl_kernel.setArg(idx++, *((cl::Image *)input->GetHandle().base));
    unit.ocl_kernel.setArg(idx++, *((cl::Image *)output->GetHandle().base));
    
    return TNN_OK;
}

std::vector<DataType> OpenCLNotLayerAcc::SupportDataType(int dims_size, BlobType blob_type) {
    return {DATA_TYPE_INT8};
}


REGISTER_OPENCL_ACC(Not, LAYER_NOT)
REGISTER_OPENCL_LAYOUT(LAYER_NOT, DATA_FORMAT_NHC4W4);

}  // namespace TNN_NS
