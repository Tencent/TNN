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

#include "tnn/device/metal/acc/metal_layer_acc.h"
#include "tnn/device/metal/acc/metal_common.h"
#include "tnn/device/metal/metal_context.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

class MetalUpsampleLayerAcc : public MetalLayerAcc {
public:
    virtual ~MetalUpsampleLayerAcc(){};
    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
    virtual Status AllocateBufferParam(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
    virtual Status Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
    virtual std::string KernelName(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
    virtual Status ComputeThreadSize(const std::vector<Blob *> &inputs,
                             const std::vector<Blob *> &outputs,
                             MTLSize &size);
    virtual Status SetKernelEncoderParam(id<MTLComputeCommandEncoder> encoder,
                                 const std::vector<Blob *> &inputs,
                                 const std::vector<Blob *> &outputs);
    virtual Status ReloadConstantBlobs(const std::vector<Blob *> &inputs, bool only_reload_shape_differ_blob = false) { return TNN_OK; }
};

Status MetalUpsampleLayerAcc::Reshape(const std::vector<Blob *> &inputs,
                                      const std::vector<Blob *> &outputs) {
    return MetalLayerAcc::Reshape(inputs, outputs);
}

Status MetalUpsampleLayerAcc::AllocateBufferParam(const std::vector<Blob *> &inputs,
                                           const std::vector<Blob *> &outputs) {
    id<MTLDevice> device = [TNNMetalDeviceImpl sharedDevice];
    UpsampleLayerParam *layer_param =
        dynamic_cast<UpsampleLayerParam *>(param_);

    auto dims_input  = inputs[0]->GetBlobDesc().dims;
    auto dims_output = outputs[0]->GetBlobDesc().dims;
    // buffer_param_
    {
        MetalUpsampleParams metal_params;
        SetDefaultMetalParams(metal_params, dims_input, dims_output);
        
        float scale_x = 0;
        float scale_y = 0;
        if (layer_param->mode == 1) {
            scale_x = (float)metal_params.input_width / (float)metal_params.output_width;
            scale_y = (float)metal_params.input_height / (float)metal_params.output_height;
        } else if (layer_param->mode == 2 || layer_param->mode == 3) {
            if (layer_param->align_corners) {
                scale_x = (metal_params.output_width > 1) ? (float)(metal_params.input_width - 1) / (metal_params.output_width - 1) : 0.f;
                scale_y = (metal_params.output_height > 1) ? (float)(metal_params.input_height - 1) / (metal_params.output_height - 1) : 0.f;
            } else {
                scale_x = (metal_params.output_width > 1) ? (float)(metal_params.input_width) / (metal_params.output_width) : 0.f;
                scale_y = (metal_params.output_height > 1) ? (float)(metal_params.input_height) / (metal_params.output_height) : 0.f;
            }
        }
        
        metal_params.scale_x = scale_x;
        metal_params.scale_y = scale_y;

        buffer_param_ =
            [device newBufferWithBytes:(const void *)(&metal_params)
                                length:sizeof(MetalUpsampleParams)
                               options:MTLResourceCPUCacheModeWriteCombined];
    }
    return TNN_OK;
}

Status MetalUpsampleLayerAcc::SetKernelEncoderParam(
                                                 id<MTLComputeCommandEncoder> encoder,
                                            const std::vector<Blob *> &inputs,
                                            const std::vector<Blob *> &outputs) {
    return MetalLayerAcc::SetKernelEncoderParam(encoder, inputs, outputs);
}

Status MetalUpsampleLayerAcc::ComputeThreadSize(const std::vector<Blob *> &inputs,
                                        const std::vector<Blob *> &outputs,
                                        MTLSize &size) {
    auto dims_output = outputs[0]->GetBlobDesc().dims;
    size = GetDefaultThreadSize(dims_output, false);
    return TNN_OK;
}

std::string MetalUpsampleLayerAcc::KernelName(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<UpsampleLayerParam *>(param_);
    if (layer_param->mode == 1) {
        // nearest align_corners=False?待确认
        return "upsample_nearest";
    } else if (layer_param->mode == 2) {
        // bilinear/linear align_corners=True
        if (layer_param->align_corners) {
            return "upsample_bilinear_align";
        } else {
            return "upsample_bilinear_noalign";
        }
    } else if (layer_param->mode == 3) {
        if (layer_param->align_corners) {
            return "upsample_cubic_align";
        } else {
            return "upsample_cubic_noalign";
        }
    } else {
        LOGE("upsample type not support!\n");
        return "";
    }
}

Status MetalUpsampleLayerAcc::Forward(const std::vector<Blob *> &inputs,
                                      const std::vector<Blob *> &outputs) {
    return MetalLayerAcc::Forward(inputs, outputs);
}

REGISTER_METAL_ACC(Upsample, LAYER_UPSAMPLE);
REGISTER_METAL_LAYOUT(LAYER_UPSAMPLE, DATA_FORMAT_NC4HW4);

} // namespace TNN_NS
