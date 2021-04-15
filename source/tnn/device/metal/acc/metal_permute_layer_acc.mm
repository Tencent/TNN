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
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

DECLARE_METAL_ACC(Permute, LAYER_PERMUTE);

const static std::map<unsigned int, std::string> kernels = {
    // NC4HW4
    {0x0231, "permute_to_nhwc"},
    {0x0213, "permute_to_nhcw"},
    {0x0312, "permute_to_nwch"},
    {0x0321, "permute_to_nwhc"},
    {0x1230, "permute_to_chwn"},
    {0x3012, "permute_to_wnch"},
    {0x0123, "permute_copy"},
    // NCHW
    //{0x4000, "permute_nchw"}
};

unsigned int GetPermuteKernelKey(DataFormat format, const std::vector<int>& orders) {
    unsigned int kid = 0;
    if (format == DATA_FORMAT_NC4HW4 && orders.size() == 4) {
        constexpr static unsigned int keys[4] = {
            0x1000, 0x0100, 0x010, 0x0001};
        kid = orders[0]*keys[0] + orders[1]*keys[1] + \
                             orders[2]*keys[2] + orders[3]*keys[3];
    }
    return kid;
}

bool hasKernelFor(Blob *input, const std::vector<int>& orders) {
    const auto blob_format = input->GetBlobDesc().data_format;
    const auto kernel_id = GetPermuteKernelKey(blob_format, orders);
    return kernels.count(kernel_id) != 0;
}

std::string GetPermuteKernel(Blob *input, const std::vector<int>& orders) {
    const auto blob_format = input->GetBlobDesc().data_format;
    /*
    // try specialized kernels first
    if (orders.size() == 4) {
        auto kernel_key = GetPermuteKernelKey(blob_format, orders);
        if (kernels.count(kernel_key) > 0)
            return kernels.at(kernel_key);
        return "permute_common4";
    } else if (orders.size() == 3) {
        auto new_orders = orders;
        new_orders.push_back(3);
        if (hasKernelFor(input, new_orders)) return GetPermuteKernel(input, new_orders);
        new_orders.clear();
        for(const auto& i : orders) {
            if (i == 2) new_orders.push_back(3);
            else new_orders.push_back(i);
        }
        new_orders.push_back(2);
        if (hasKernelFor(input, new_orders)) return GetPermuteKernel(input, new_orders);
    } else if (orders.size() == 2) {
        auto new_orders = orders;
        new_orders.push_back(2);
        new_orders.push_back(3);
        if (hasKernelFor(input, new_orders)) return GetPermuteKernel(input, new_orders);
        new_orders = orders;
        new_orders.push_back(3);
        new_orders.push_back(2);
        if (hasKernelFor(input, new_orders)) return GetPermuteKernel(input, new_orders);
    }
    */
    return "permute_common";
}

Status MetalPermuteLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<PermuteLayerParam *>(param_);
    if (GetPermuteKernel(inputs[0], layer_param->orders) == "") {
        return Status(TNNERR_PARAM_ERR, "permute orders not supported!");
    }
    return MetalLayerAcc::Reshape(inputs, outputs);
}

Status MetalPermuteLayerAcc::AllocateBufferParam(const std::vector<Blob *> &inputs,
                                                 const std::vector<Blob *> &outputs) {
    id<MTLDevice> device = [TNNMetalDeviceImpl sharedDevice];
    PermuteLayerParam *params = dynamic_cast<PermuteLayerParam *>(param_);
    auto dims_input      = inputs[0]->GetBlobDesc().dims;
    auto dims_output     = outputs[0]->GetBlobDesc().dims;
    // buffer_param_
    {
        MetalDynamicPermuteParams metal_params;
        if (dims_input.size() > MAX_DIM_COUNT) {
            LOGE("too many dimensions in inputs to permute layer!");
            return Status(TNNERR_PARAM_ERR, "too many dimensions in inputs to permute layer!");
        }

        const int dim_count = dims_input.size();
        metal_params.dim_count = dim_count;
        const int input_slice  = UP_DIV(dims_input[1], 4);
        const int output_slice = UP_DIV(dims_output[1], 4);
        metal_params.input_slice = input_slice;
        metal_params.output_slice = output_slice;
        metal_params.input_batch = dims_input[0];
        metal_params.batch = dims_output[0];

        int input_strides[MAX_DIM_COUNT] = {0};
        int stride = 1;
        int input_size = 1;
        int output_size = 1;
        for(int i=dim_count-1; i>=0; --i) {
            input_strides[i] = stride;
            if (i == 1) {
                stride *= input_slice;
            } else {
                stride *= dims_input[i];
            }
            metal_params.input_sizes[i] = dims_input[i];
            metal_params.output_sizes[i] = dims_output[i];
            if (i > 1) {
                input_size *= dims_input[i];
                output_size *= dims_output[i];
            }
        }
        metal_params.input_size = input_size;
        metal_params.output_size = output_size;

        int output_strides[MAX_DIM_COUNT] = {0};
        const auto& orders = params->orders;
        for(int i=0; i<dim_count; ++i) {
            output_strides[i] = input_strides[orders[i]];
            if (orders[i] == 1) {
                metal_params.channel_dim = i;
            }
            if (i == 1) {
                metal_params.channel_dim_size = dims_input[orders[i]];
            }
            metal_params.strides[i] = output_strides[i];
        }

        buffer_param_     = [device newBufferWithBytes:(const void *)(&metal_params)
                                            length:sizeof(metal_params)
                                           options:MTLResourceCPUCacheModeWriteCombined];
    }
    return TNN_OK;
}

Status MetalPermuteLayerAcc::SetKernelEncoderParam(
                                                 id<MTLComputeCommandEncoder> encoder,
                                            const std::vector<Blob *> &inputs,
                                            const std::vector<Blob *> &outputs) {
    return MetalLayerAcc::SetKernelEncoderParam(encoder, inputs, outputs);
}

std::string MetalPermuteLayerAcc::KernelName(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<PermuteLayerParam *>(param_);
    const auto &kernel_name = GetPermuteKernel(inputs[0], layer_param->orders);
    return kernel_name;
}

Status MetalPermuteLayerAcc::ComputeThreadSize(const std::vector<Blob *> &inputs,
                                        const std::vector<Blob *> &outputs,
                                        MTLSize &size) {
    auto dims_output = outputs[0]->GetBlobDesc().dims;
    size = MTLSizeMake(DimsFunctionUtils::GetDimProduct(dims_output, 2),
                        UP_DIV(dims_output[1], 4),
                        dims_output[0]);
    return TNN_OK;
}

Status MetalPermuteLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return MetalLayerAcc::Forward(inputs, outputs);
}

REGISTER_METAL_ACC(Permute, LAYER_PERMUTE);
REGISTER_METAL_LAYOUT(LAYER_PERMUTE, DATA_FORMAT_NC4HW4);

} // namespace TNN_NS
