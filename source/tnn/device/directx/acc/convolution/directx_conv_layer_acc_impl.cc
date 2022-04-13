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

#include "tnn/device/directx/acc/convolution/directx_conv_layer_acc_impl.h"
#include "tnn/utils/dims_utils.h"
#include "tnn/utils/string_utils_inner.h"

#include "tnn/device/directx/directx_util.h"

namespace TNN_NS {
namespace directx {

DirectXConvLayerAccImpl::DirectXConvLayerAccImpl() {
    conv_type_ = CT_CONV_COMMON;
}

Status DirectXConvLayerAccImpl::Init(Context *context, LayerParam *param, LayerResource *resource,
                                    const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    Status ret = DirectXLayerAcc::Init(context, param, resource, inputs, outputs);
    RETURN_ON_NEQ(ret, TNN_OK);

    ConvLayerParam *conv_param = dynamic_cast<ConvLayerParam *>(param);
    if (nullptr == conv_param) {
        LOGE("invalid conv param!\n");
        return Status(TNNERR_NULL_PARAM, "invalid conv param");
    }

    // interpreter conv 2d paramm
    conv_params_.kernel_w        = conv_param->kernels[0];
    conv_params_.kernel_h        = conv_param->kernels[1];
    conv_params_.pad_w           = conv_param->pads[0];
    conv_params_.pad_h           = conv_param->pads[2];
    conv_params_.stride_w        = conv_param->strides[0];
    conv_params_.stride_h        = conv_param->strides[1];
    conv_params_.dilation_w      = conv_param->dialations[0];
    conv_params_.dilation_h      = conv_param->dialations[1];
    conv_params_.pad_type        = conv_param->pad_type;
    conv_params_.group           = conv_param->group;
    conv_params_.has_bias        = conv_param->bias;
    conv_params_.activation_type = conv_param->activation_type;

    conv_params_.input_channel  = DimsFunctionUtils::GetDim(inputs[0]->GetBlobDesc().dims, 1);
    conv_params_.output_channel = DimsFunctionUtils::GetDim(outputs[0]->GetBlobDesc().dims, 1);

    if ((conv_params_.group <= 0 || conv_params_.input_channel % conv_params_.group != 0)) {
        LOGE("invalid group size in Conv layer!\n");
        return Status(TNNERR_LAYER_ERR, "invalid group size in Conv layer");
    }

    use_buffer_ = false;

    return TNN_OK;
}

DirectXConvLayerAccImpl::~DirectXConvLayerAccImpl() {}

Status DirectXConvLayerAccImpl::AllocateWeightsBias(LayerResource *resource) {
    Status ret                       = TNN_OK;
    ConvLayerResource *conv_resource = dynamic_cast<ConvLayerResource *>(resource);
    if (nullptr == conv_resource) {
        LOGE("invalid conv resource!\n");
        return Status(TNNERR_NULL_PARAM, "invalid conv resource");
    }

    // convert weights
    float *weights_data_ptr = conv_resource->filter_handle.force_to<float *>();
    std::shared_ptr<float> float_weight = std::shared_ptr<float>(weights_data_ptr, [](float *){});

    if (conv_resource->filter_handle.GetDataType() != DATA_TYPE_FLOAT) {
        float_weight = std::move(GetFloatFromRawBuffer(conv_resource->filter_handle));
        if (float_weight == nullptr) {
            return Status(TNNERR_DX_ACC_INIT_ERR, "convert weights to float failed");
        }
    }
    ret = ConvertWeights(float_weight.get());
    RETURN_ON_NEQ(ret, TNN_OK);

    // convert bias
    ret = ConvertChannelWeights(conv_resource->bias_handle, bias_, conv_params_.output_channel,
                                conv_params_.has_bias, false, use_buffer_);
    return ret;
}

// convert weights will copy data to buffer, then:
// if use clBuffer weigths for kernel, will convert buffer to buffer with target format.
// if use clImage weights for kernel, will convert buffer to image with target format.
Status DirectXConvLayerAccImpl::ConvertWeights(float *weights_data_ptr) {

    float *wdata_ptr              = weights_data_ptr;
    std::shared_ptr<float> weights_data_trans;
    // special for group conv
    if (CT_CONV_COMMON == conv_type_ && conv_params_.group > 1) {
        int element_size =
            conv_params_.output_channel * conv_params_.input_channel * conv_params_.kernel_h * conv_params_.kernel_w;
        weights_data_trans.reset(new float[element_size], [](float *p) { delete[] p; });

        // Convert from GOIHW to OIHW
        GROUP_PADDING<float, int>(weights_data_ptr, weights_data_trans.get(), conv_params_.group,
                                  conv_params_.output_channel, conv_params_.input_channel, conv_params_.kernel_h,
                                  conv_params_.kernel_w, GOIHW);
        wdata_ptr = weights_data_trans.get();
    }

    // copy weights data into Buffer
    DimsVector filter_shape;
    if (CT_CONV_DEPTHWISE == conv_type_) {
        filter_shape = {1, conv_params_.output_channel, conv_params_.kernel_h, conv_params_.kernel_w};
    } else {
        filter_shape = {conv_params_.output_channel, conv_params_.input_channel, conv_params_.kernel_h, conv_params_.kernel_w};
    }

    if (precision_ == PRECISION_LOW) {
        LOGE("FP16 Weigths not supported now.");
        return Status(TNNERR_DX_ACC_INIT_ERR, "FP16 weights not supported now.");
    }


    if (use_buffer_) {
        auto dx_mem = DirectXMemory::CreateBufferMemoryFromHost(wdata_ptr, filter_shape, DATA_TYPE_FLOAT, DATA_FORMAT_NCHW);
        if (!dx_mem) {
            LOGE("CreateBufferMemoryFromHost failed\n");
            return Status(TNNERR_DX_BUFFER_ALOCATE_ERR, "create directx buffer memory failed.");
        }

        weights_ = std::move(dx_mem);

        // TODO pad channel.
        // convert to packC4 weights 
        DimsVector filter_buffershape;
        if (CT_CONV_DEPTHWISE == conv_type_) {
            filter_buffershape = {1, ROUND_UP(conv_params_.output_channel, 4),
                                  conv_params_.kernel_h, conv_params_.kernel_w};
        } else {
            filter_buffershape = {ROUND_UP(conv_params_.output_channel, 4), ROUND_UP(conv_params_.input_channel, 4),
                                  conv_params_.kernel_h, conv_params_.kernel_w};
        }

    } else {
        // create weights use texture2d
        DimsVector filter_imageshape;
        if (CT_CONV_DEPTHWISE == conv_type_) {
            filter_imageshape = {conv_params_.kernel_w * conv_params_.kernel_h,
                                 (int)(UP_DIV(conv_params_.output_channel, 4))};  // {w,h}
        } else {
            filter_imageshape = {conv_params_.input_channel, (int)(UP_DIV(conv_params_.output_channel, 4) *
                                 conv_params_.kernel_w * conv_params_.kernel_h)};
        }

        auto dx_mem = DirectXMemory::CreateTextureMemoryFromHost(nullptr, filter_shape, filter_imageshape[0], filter_imageshape[1], DATA_TYPE_FLOAT, DATA_FORMAT_NHC4W4);
        if (!dx_mem) {
            LOGE("CreateTextureMemoryFromHost failed\n");
            return Status(TNNERR_DX_TEXTURE_ALOCATE_ERR, "create directx texture memory failed.");
        }

        Status ret = UpdateConv2DFilterTexture2D(wdata_ptr, filter_shape, filter_imageshape[0], filter_imageshape[1], dx_mem);
        RETURN_ON_NEQ(ret, TNN_OK);

        weights_ = std::move(dx_mem);
    }

    return TNN_OK;
}

#if TNN_PROFILE
double DirectXConvLayerAccImpl::GetFlops() {
    return 2.0 * DimsVectorUtils::Count(output_dims_) * input_dims_[1] / conv_params_.group * conv_params_.kernel_w *
           conv_params_.kernel_h;
}
#endif

}  // namespace directx 
}  // namespace TNN_NS
