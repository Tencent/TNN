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
#include "tnn/device/opencl/imagebuffer_convertor.h"

namespace TNN_NS {

class OpenCLUpsampleLayerAcc : public OpenCLLayerAcc {
public:
    virtual Status Init(Context *context, LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                        const std::vector<Blob *> &outputs) override;

    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;

    virtual Status ReloadConstantBlobs(const std::vector<Blob *> &inputs, bool only_reload_shape_differ_blob = false) override { return TNN_OK; }
};

static std::vector<uint32_t> UpsampleLocalWS3D(std::vector<uint32_t> &gws, const uint32_t max_workgroup_size) {
    uint32_t compute_units = OpenCLRuntime::GetInstance()->DeviceComputeUnits();
    GpuType gpu_type       = OpenCLRuntime::GetInstance()->GetGpuInfo().type;
    std::vector<uint32_t> lws(3, 0);
    if (gpu_type == GpuType::ADRENO) {
        lws[0] = 4;
        lws[1] = gcd(gcd(max_workgroup_size / 16, compute_units * 4), gws[1]);
        lws[2] = 4;
    } else {
        lws[0] = 4;
        lws[1] = max_workgroup_size / 16;
        lws[2] = 4;
    }
    return lws;
}

Status OpenCLUpsampleLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                    const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init Upsample Acc\n");
    Status ret = OpenCLLayerAcc::Init(context, param, resource, inputs, outputs);
    CHECK_TNN_OK(ret)

    op_name_ = "Upsample";

    UpsampleLayerParam *upsample_param = dynamic_cast<UpsampleLayerParam *>(param);
    if (!upsample_param) {
        LOGE("Error: layer param is null\n");
        return Status(TNNERR_MODEL_ERR, "Error: layer param is null");
    }

    // create kernel
    std::string kernel_name;
    if (upsample_param->mode == 1) {  // nearst
        LOGD("build nearest\n");
        kernel_name = "Nearest";
    } else if (upsample_param->mode == 2) {  // bilinear
        if (upsample_param->align_corners) {
            LOGD("build bilinear with aligned corners\n");
            kernel_name = "BilinearAlignCorners";
        } else {
            LOGD("build bilinear\n");
            kernel_name = "Bilinear";
        }
    } else if (upsample_param->mode == 3) {  // cubic
        if (upsample_param->align_corners) {
            LOGD("build cubic with aligned corners\n");
            kernel_name = "CubicAlignCorners";
        } else {
            LOGD("build cubic\n");
            kernel_name = "Cubic";
        }
    } else {
        LOGE("Not support Upsample type: %d\n", upsample_param->mode);
        return Status(TNNERR_OPENCL_ACC_INIT_ERROR, "invalid upsample mode");
    }
    if (run_3d_ndrange_) {
        kernel_name += "GS3D";
    }

    ret = CreateExecuteUnit(execute_units_[0], "upsample", kernel_name, build_options_);
    if (ret != TNN_OK) {
        LOGE("create execute unit failed!\n");
        return ret;
    }

    return TNN_OK;
}

Status OpenCLUpsampleLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Upsample Acc Reshape\n");
    std::vector<Blob *> reshape_inputs(inputs.begin(), inputs.begin() + 1);
    Status ret = OpenCLLayerAcc::Reshape(reshape_inputs, outputs);
    CHECK_TNN_OK(ret)

    UpsampleLayerParam *upsample_param = dynamic_cast<UpsampleLayerParam *>(param_);
    if (!upsample_param) {
        LOGE("Error: layer param is null\n");
        return Status(TNNERR_MODEL_ERR, "Error: layer param is null");
    }

    auto input  = inputs[0];
    auto output = outputs[0];

    auto input_dims  = input->GetBlobDesc().dims;
    auto output_dims = output->GetBlobDesc().dims;

    const int batch        = DimsFunctionUtils::GetDim(input_dims, 0);
    const int channels     = DimsFunctionUtils::GetDim(input_dims, 1);
    const int input_height = DimsFunctionUtils::GetDim(input_dims, 2);
    const int input_width  = DimsFunctionUtils::GetDim(input_dims, 3);

    const int output_height = DimsFunctionUtils::GetDim(output_dims, 2);
    const int output_width  = DimsFunctionUtils::GetDim(output_dims, 3);

    const int channel_blocks = UP_DIV(channels, 4);

    float height_scale;
    float width_scale;
    if ((upsample_param->mode == 2 || upsample_param->mode == 3) && upsample_param->align_corners) {
        height_scale = (float)(input_height - 1) / (float)(output_height - 1);
        width_scale  = (float)(input_width - 1) / (float)(output_width - 1);
    } else {
        height_scale = (float)input_height / (float)output_height;
        width_scale  = (float)input_width / (float)output_width;
    }

    uint32_t idx = 0;
    if (run_3d_ndrange_) {
        execute_units_[0].global_work_size = {static_cast<uint32_t>(output_width),
                                            static_cast<uint32_t>(channel_blocks),
                                            static_cast<uint32_t>(batch * output_height)};
        execute_units_[0].local_work_size =
            UpsampleLocalWS3D(execute_units_[0].global_work_size, execute_units_[0].workgroupsize_max);
        for (auto gws : execute_units_[0].global_work_size) {
            execute_units_[0].ocl_kernel.setArg(idx++, gws);
        }
    } else {
        idx = SetExecuteUnit2DSizeInfoDefault(execute_units_[0], output_dims);
    }

    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)input->GetHandle().base));
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)output->GetHandle().base));
    execute_units_[0].ocl_kernel.setArg(idx++, height_scale);
    execute_units_[0].ocl_kernel.setArg(idx++, width_scale);
    execute_units_[0].ocl_kernel.setArg(idx++, static_cast<int32_t>(input_height));
    execute_units_[0].ocl_kernel.setArg(idx++, static_cast<int32_t>(input_width));
    execute_units_[0].ocl_kernel.setArg(idx++, static_cast<int32_t>(output_height));
    if (!run_3d_ndrange_) {
        execute_units_[0].ocl_kernel.setArg(idx++, static_cast<int32_t>(output_width));
    }

    return TNN_OK;
}

REGISTER_OPENCL_ACC(Upsample, LAYER_UPSAMPLE)
REGISTER_OPENCL_LAYOUT(LAYER_UPSAMPLE, DATA_FORMAT_NHC4W4);

}  // namespace TNN_NS
