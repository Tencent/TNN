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

class OpenCLArgMaxOrMinLayerAcc : public OpenCLLayerAcc {
public:
    virtual Status Init(Context *context, LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                        const std::vector<Blob *> &outputs) override;

    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;

private:
    virtual std::vector<DataType> SupportDataType(int dims_size, BlobType blob_type) override;
};

Status OpenCLArgMaxOrMinLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                       const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init ArgMaxOrMin Acc\n");
    Status ret = OpenCLLayerAcc::Init(context, param, resource, inputs, outputs);
    CHECK_TNN_OK(ret)

    run_3d_ndrange_ = false;
    op_name_        = "ArgMaxOrMin";

    ArgMaxOrMinLayerParam *arg_max_or_min_param = dynamic_cast<ArgMaxOrMinLayerParam *>(param);
    if (!arg_max_or_min_param) {
        LOGE("Error: layer param is null\n");
        return Status(TNNERR_MODEL_ERR, "Error: layer param is null");
    }

    // create kernel
    std::string kernel_name;
    if (0 == arg_max_or_min_param->axis) {
        kernel_name = "ArgOpN";
    } else if (1 == arg_max_or_min_param->axis) {
        kernel_name = "ArgOpC";
    } else if (2 == arg_max_or_min_param->axis) {
        kernel_name = "ArgOpH";
    } else if (3 == arg_max_or_min_param->axis) {
        kernel_name = "ArgOpW";
    } else {
        LOGE("not support axis = %d in argmax/min yet!\n", arg_max_or_min_param->axis);
        return Status(TNNERR_OPENCL_ACC_INIT_ERROR, "invalid argmax/min axis");
    }

    std::set<std::string> build_options;
    std::string compute, operator_func;
    if (0 == arg_max_or_min_param->mode) {
        compute         = "MinOp";
        operator_func   = "min";
    } else if (1 == arg_max_or_min_param->mode) {
        compute         = "MaxOp";
        operator_func   = "max";
    } else {
        LOGE("not support mode = %d in argmax/min yet!\n", arg_max_or_min_param->mode);
        return Status(TNNERR_OPENCL_ACC_INIT_ERROR, "invalid argmax/min mode");
    }

    build_options.emplace(" -DOPERATOR=" + operator_func + " -DBINARY_OPERATOR=" + compute);
    build_options.insert(build_options_.begin(), build_options_.end());

    ret = CreateExecuteUnit(execute_units_[0], "arg", kernel_name, build_options);
    if (ret != TNN_OK) {
        LOGE("create execute unit failed!\n");
        return ret;
    }

    return TNN_OK;
}

Status OpenCLArgMaxOrMinLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("ArgMaxOrMin Acc Reshape\n");
    Status ret = OpenCLLayerAcc::Reshape(inputs, outputs);
    CHECK_TNN_OK(ret)

    ArgMaxOrMinLayerParam *arg_max_or_min_param = dynamic_cast<ArgMaxOrMinLayerParam *>(param_);
    if (!arg_max_or_min_param) {
        LOGE("Error: layer param is null\n");
        return Status(TNNERR_MODEL_ERR, "Error: layer param is null");
    }
    auto output_dims        = outputs[0]->GetBlobDesc().dims;
    auto input_dims         = inputs[0]->GetBlobDesc().dims;
    int axis                = arg_max_or_min_param->axis;
    int select_last_index   = arg_max_or_min_param->select_last_index;

    if (select_last_index != 0) {
        LOGE("Error: select_last_index is not supported for now\n");
        return Status(TNNERR_MODEL_ERR, "Error: select_last_index in ArgMax/ArgMin failed");
    }

    uint32_t idx = SetExecuteUnit2DSizeInfoDefault(execute_units_[0], output_dims);
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)inputs[0]->GetHandle().base));
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)outputs[0]->GetHandle().base));
    // only support up to 4 dims for now
    for (int i = 0; i < 4; i++) {
        if (i != axis)
            execute_units_[0].ocl_kernel.setArg(idx++, DimsFunctionUtils::GetDim(output_dims, i));
    }
    execute_units_[0].ocl_kernel.setArg(idx++, DimsFunctionUtils::GetDim(input_dims, axis));
    // channel arg op
    if (axis == 1) {
        execute_units_[0].ocl_kernel.setArg(idx++, UP_DIV(DimsFunctionUtils::GetDim(input_dims, axis), 4));
    }

    return TNN_OK;
}

std::vector<DataType> OpenCLArgMaxOrMinLayerAcc::SupportDataType(int dims_size, BlobType blob_type) {
    if (blob_type == BLOB_INPUT) {
        return {DATA_TYPE_FLOAT, DATA_TYPE_HALF};
    } else {
        return {DATA_TYPE_INT32};
    }
}

REGISTER_OPENCL_ACC(ArgMaxOrMin, LAYER_ARG_MAX_OR_MIN)
REGISTER_OPENCL_LAYOUT(LAYER_ARG_MAX_OR_MIN, DATA_FORMAT_NHC4W4);

}  // namespace TNN_NS