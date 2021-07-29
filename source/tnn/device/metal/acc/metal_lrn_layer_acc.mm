//  Copyright © 2019 tencent. All rights reserved.

#include "tnn/device/metal/acc/metal_common.h"
#include "tnn/device/metal/acc/metal_layer_acc.h"
#include "tnn/device/metal/metal_context.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

DECLARE_METAL_ACC(LRN, LAYER_LRN);

Status MetalLRNLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return MetalLayerAcc::Reshape(inputs, outputs);
}

Status MetalLRNLayerAcc::AllocateBufferParam(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {

    id<MTLDevice> device = [TNNMetalDeviceImpl sharedDevice];
    auto layer_param     = dynamic_cast<LRNLayerParam *>(param_);
    if (!layer_param) {
        LOGE("Error: layer param is invalid\n");
        return Status(TNNERR_MODEL_ERR, "Error: layer param is invalid");
    }

    auto dims_input  = inputs[0]->GetBlobDesc().dims;
    auto dims_output = outputs[0]->GetBlobDesc().dims;
    // buffer_param_
    {
        MetalLRNParams metal_params;
        SetDefaultMetalParams(metal_params, dims_input, dims_output);
        metal_params.input_channel = dims_input[1];
        metal_params.alpha         = layer_param->alpha;
        metal_params.beta          = layer_param->beta;
        metal_params.bias          = layer_param->bias;
        metal_params.size          = layer_param->size;

        buffer_param_ = [device newBufferWithBytes:(const void *)(&metal_params)
                                            length:sizeof(MetalLRNParams)
                                           options:MTLResourceCPUCacheModeWriteCombined];
    }
    return TNN_OK;
}

std::string MetalLRNLayerAcc::KernelName(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return "lrn_across_channel";
}

Status MetalLRNLayerAcc::SetKernelEncoderParam(
                                                 id<MTLComputeCommandEncoder> encoder,
                                            const std::vector<Blob *> &inputs,
                                            const std::vector<Blob *> &outputs) {
    return MetalLayerAcc::SetKernelEncoderParam(encoder, inputs, outputs);
}

Status MetalLRNLayerAcc::Forward(const std::vector<Blob *> &inputs,
                                   const std::vector<Blob *> &outputs) {
    return MetalLayerAcc::Forward(inputs, outputs);
}

Status MetalLRNLayerAcc::ComputeThreadSize(const std::vector<Blob *> &inputs,
                                        const std::vector<Blob *> &outputs,
                                        MTLSize &size) {
    auto dims_output = outputs[0]->GetBlobDesc().dims;
    size = GetDefaultThreadSize(dims_output, false);
    return TNN_OK;
}

REGISTER_METAL_ACC(LRN, LAYER_LRN);
REGISTER_METAL_LAYOUT(LAYER_LRN, DATA_FORMAT_NC4HW4);

} // namespace TNN_NS
