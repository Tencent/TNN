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

#include "tnn/device/metal/acc/metal_common.h"
#include "tnn/device/metal/acc/metal_layer_acc.h"
#include "tnn/device/metal/metal_context.h"
#include "tnn/utils/dims_vector_utils.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/half_utils.h"
#include "tnn/utils/pribox_generator_utils.h"

namespace TNN_NS {

// @brief conv layer metal acc
class MetalPriorBoxLayerAcc : public MetalLayerAcc {
public:
    virtual ~MetalPriorBoxLayerAcc();
    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
    virtual std::string KernelName(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
    virtual Status SetKernelEncoderParam(id<MTLComputeCommandEncoder> encoder,
                                         const std::vector<Blob *> &inputs,
                                         const std::vector<Blob *> &outputs);

protected:
    id<MTLBuffer> buffer_priorbox_ = nil;
    DimsVector buffer_priorbox_shape_ = {};
};

MetalPriorBoxLayerAcc::~MetalPriorBoxLayerAcc(){
}

Status MetalPriorBoxLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto param = dynamic_cast<PriorBoxLayerParam *>(param_);
    if (!param) {
        return Status(TNNERR_MODEL_ERR, "Error: layer param is nil");
    }
    
    auto dims_output = outputs[0]->GetBlobDesc().dims;
    
    if (!buffer_priorbox_ || !DimsVectorUtils::Equal(buffer_priorbox_shape_, dims_output)) {
        auto priorbox = GeneratePriorBox(inputs, outputs, param);
        RawBuffer raw_prior_box((int)priorbox.size()*sizeof(float), (char *)priorbox.data());
        
        Status status = TNN_OK;
        buffer_priorbox_ = AllocatePackedNC4HW4MetalBufferFormRawBuffer(raw_prior_box, dims_output, 1, status);
        RETURN_ON_NEQ(status, TNN_OK);
    }
    
    return MetalLayerAcc::Reshape(inputs, outputs);
}

std::string MetalPriorBoxLayerAcc::KernelName(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return "copy_buffer_2_buffer";
}

Status MetalPriorBoxLayerAcc::SetKernelEncoderParam(
                                                id<MTLComputeCommandEncoder> encoder,
                                                const std::vector<Blob *> &inputs,
                                                const std::vector<Blob *> &outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];
    
    [encoder setBuffer:buffer_priorbox_
                offset:0
               atIndex:0];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(void *)output->GetHandle().base
                offset:(NSUInteger)output->GetHandle().bytes_offset
               atIndex:1];
    [encoder setBuffer:buffer_param_ offset:0 atIndex:2];
    
    return TNN_OK;
}

REGISTER_METAL_ACC(PriorBox, LAYER_PRIOR_BOX);

} // namespace TNN_NS
