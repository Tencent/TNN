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

// DECLARE_OPENCL_ACC(Cast);
class OpenCLCastLayerAcc : public OpenCLLayerAcc {
public:
    virtual Status Init(Context *context, LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                        const std::vector<Blob *> &outputs) override;

    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;

private:
    virtual std::vector<DataType> SupportDataType(int dims_size, BlobType blob_type) override;
};

Status OpenCLCastLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                               const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init Cast Acc\n");
    Status ret = OpenCLLayerAcc::Init(context, param, resource, inputs, outputs);
    CHECK_TNN_OK(ret)

    auto input_data_type  = inputs[0]->GetBlobDesc().data_type;
    auto output_data_type = outputs[0]->GetBlobDesc().data_type;
    execute_units_.resize(1);
    
    

    if(input_data_type == output_data_type) {
        ret         = CreateExecuteUnit(execute_units_[0], "copy", "CopyImage");
        if (ret != TNN_OK) {
            return ret;
        }
    } else if(input_data_type == DATA_TYPE_INT32 && (output_data_type == DATA_TYPE_FLOAT || output_data_type == DATA_TYPE_HALF)) {
        std::set<std::string> build_options;
        if(context->GetPrecision() != PRECISION_HIGH) {
            build_options.emplace(std::string(" -DCONVERT=") + "convert_half4");
        } else {
            build_options.emplace(std::string(" -DCONVERT=") + "convert_float4");
        }
        ret         = CreateExecuteUnit(execute_units_[0], "cast_int32_to_float", "CastIntToFloat", build_options);
        if(ret != TNN_OK) {
            return ret;
        }
    } else {
        return Status(TNNERR_PARAM_ERR, "cast not support");        
    }

    return TNN_OK;
}

Status OpenCLCastLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Cast Acc Reshape\n");
    Status ret = OpenCLLayerAcc::Reshape(inputs, outputs);
    CHECK_TNN_OK(ret)

    auto input  = inputs[0];
    auto output = outputs[0];

    auto input_dims  = input->GetBlobDesc().dims;
    auto output_dims = output->GetBlobDesc().dims;

    const int batch         = DimsFunctionUtils::GetDim(output_dims, 0);
    const int channels      = DimsFunctionUtils::GetDim(output_dims, 1);
    const int output_height = DimsFunctionUtils::GetDim(output_dims, 2);
    const int output_width  = DimsFunctionUtils::GetDim(output_dims, 3);

    int inputWH[]      = {DimsFunctionUtils::GetDim(input_dims, 3), DimsFunctionUtils::GetDim(input_dims, 2)};
    int inputOffset[]  = {0, 0, 0, 0};
    int outputOffset[] = {0, 0, 0, 0};
    int outputWH[] = {output_width, output_height};

    auto &unit = execute_units_[0];
    int idx    = SetExecuteUnit2DSizeInfoDefault(unit, output_dims);
    unit.ocl_kernel.setArg(idx++, *((cl::Image *)inputs[0]->GetHandle().base));
    unit.ocl_kernel.setArg(idx++, *((cl::Image *)output->GetHandle().base));
    unit.ocl_kernel.setArg(idx++, inputOffset);
    unit.ocl_kernel.setArg(idx++, outputOffset);
    unit.ocl_kernel.setArg(idx++, inputWH);
    unit.ocl_kernel.setArg(idx++, outputWH);
    unit.ocl_kernel.setArg(idx++, outputWH);
    
    return TNN_OK;
}

std::vector<DataType> OpenCLCastLayerAcc::SupportDataType(int dims_size, BlobType blob_type) {
    return {DATA_TYPE_FLOAT, DATA_TYPE_HALF, DATA_TYPE_INT32};
}


REGISTER_OPENCL_ACC(Cast, LAYER_CAST)
REGISTER_OPENCL_LAYOUT(LAYER_CAST, DATA_FORMAT_NHC4W4);

}  // namespace TNN_NS
