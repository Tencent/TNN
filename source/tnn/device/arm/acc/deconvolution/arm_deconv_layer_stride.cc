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

#include "tnn/device/arm/acc/deconvolution/arm_deconv_layer_stride.h"

#include <memory>

#include "tnn/device/arm/acc/convolution/arm_conv_int8_layer_common.h"
#include "tnn/device/arm/acc/convolution/arm_conv_layer_1x1.h"
#include "tnn/device/arm/acc/convolution/arm_conv_layer_3x3.h"
#include "tnn/device/arm/acc/convolution/arm_conv_layer_common.h"
#include "tnn/device/arm/arm_common.h"
#include "tnn/interpreter/raw_buffer.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

bool ArmDeconvLayerStride::isPrefered(ConvLayerParam *param, const std::vector<Blob *> &inputs,
                                      const std::vector<Blob *> &outputs) {
    if (!param)
        return false;
    // return false;
    // select stride
    return param->group == 1 && param->strides[0] > 1 && param->strides[1] > 1 && param->dialations[0] == 1 &&
           param->dialations[1] == 1;
}

Status ArmDeconvLayerStride::Init(Context *context, LayerParam *param, LayerResource *resource,
                                  const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    ConvLayerParam *conv_param = dynamic_cast<ConvLayerParam *>(param);
    CHECK_PARAM_NULL(conv_param);
    ConvLayerResource *conv_res = dynamic_cast<ConvLayerResource *>(resource);
    CHECK_PARAM_NULL(conv_res);

    RETURN_ON_NEQ(ArmLayerAcc::Init(context, param, resource, inputs, outputs), TNN_OK);
    // step0: split param into groups
    conv_units_.clear();
    RETURN_ON_NEQ(CreateStrideConvUnit(), TNN_OK);

    // step1: set split blob desc
    RETURN_ON_NEQ(SetSplitBlobDesc(inputs[0]), TNN_OK);

    // step2: set split resource, crop and transpose
    RETURN_ON_NEQ(SplitResource(), TNN_OK);

    // step3: create conv impl accroding to split params
    for (auto &unit : conv_units_) {
        std::vector<Blob *> local_outputs;
        local_outputs.emplace_back(unit.blob.get());
        std::shared_ptr<ArmLayerAcc> tmp_acc = nullptr;
        if (inputs[0]->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
            CreateImpFP(inputs, local_outputs, unit.param.get(), tmp_acc);
        } else {
            return Status(TNNERR_LAYER_ERR, "Error: stride conv not support data type");
        }

        CHECK_PARAM_NULL(tmp_acc);
        RETURN_ON_NEQ(tmp_acc->Init(context_, unit.param.get(), unit.resource.get(), inputs, local_outputs), TNN_OK);

        unit.conv_acc_impl = tmp_acc;

        // release resource, have been set into unit conv impl
        unit.resource.reset();
    }

    return TNN_OK;
}

ArmDeconvLayerStride::~ArmDeconvLayerStride() {}

Status ArmDeconvLayerStride::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

/*
get different impl based on conv params
ArmDeconvLayerCommon always as the last solution
bfp16 impl included in fp impl
*/
void ArmDeconvLayerStride::CreateImpFP(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs,
                                       LayerParam *param, std::shared_ptr<ArmLayerAcc> &conv_acc_impl) {
    if (ArmConvLayer3x3::isPrefered(dynamic_cast<ConvLayerParam *>(param_), inputs, outputs)) {
        if (!dynamic_cast<ArmConvLayer3x3 *>(conv_acc_impl.get())) {
            conv_acc_impl = std::make_shared<ArmConvLayer3x3>();
        }
    } else if (ArmConvLayer1x1::isPrefered(dynamic_cast<ConvLayerParam *>(param_), inputs, outputs)) {
        if (!dynamic_cast<ArmConvLayer1x1 *>(conv_acc_impl.get())) {
            conv_acc_impl = std::make_shared<ArmConvLayer1x1>();
        }
    }

    if (!conv_acc_impl) {
        conv_acc_impl = std::make_shared<ArmConvLayerCommon>();
    }
}

Status ArmDeconvLayerStride::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    RawBuffer blob_buf;

    RETURN_ON_NEQ(SetSplitBlobHandle(outputs[0], blob_buf), TNN_OK);
    // step0: forward conv queue
    for (auto &unit : conv_units_) {
        std::vector<Blob *> local_outputs;
        local_outputs.emplace_back(unit.blob.get());
        CHECK_PARAM_NULL(unit.conv_acc_impl.get());
        RETURN_ON_NEQ(unit.conv_acc_impl->DoForward(inputs, local_outputs), TNN_OK);
    }

    // step1: copy stride convs into one output
    CopyOutputSplitBlob(outputs[0]);

    return TNN_OK;
}

Status ArmDeconvLayerStride::CreateStrideConvUnit() {
    auto param = dynamic_cast<ConvLayerParam *>(param_);
    CHECK_PARAM_NULL(param);
    auto sy = param->strides[1];
    auto sx = param->strides[0];
    auto ky = param->kernels[1];
    auto kx = param->kernels[0];

    for (int y = 0; y < sy; y++) {
        if (y >= ky)
            continue;
        int kc_y = 1 + (ky - y - 1) / sy;
        for (int x = 0; x < sx; x++) {
            if (x >= kx)
                continue;
            int kc_x = 1 + (kx - x - 1) / sx;

            ConvUnit conv_unit;
            auto stride_conv_param            = new ConvLayerParam();
            auto stride_conv_resource         = new ConvLayerResource();
            stride_conv_param->strides        = {1, 1};
            stride_conv_param->kernels        = {kc_x, kc_y};
            stride_conv_param->pad_type       = -1;
            stride_conv_param->pads           = {kc_x - 1, kc_x - 1, kc_y - 1, kc_y - 1};
            stride_conv_param->dialations     = {1, 1};
            stride_conv_param->input_channel  = param->input_channel;
            stride_conv_param->output_channel = param->output_channel;
            conv_unit.param = std::shared_ptr<ConvLayerParam>(stride_conv_param);
            conv_unit.resource = std::shared_ptr<ConvLayerResource>(stride_conv_resource);
            conv_unit.y_offset = y;
            conv_unit.x_offset = x;
            conv_unit.kc_y = kc_y;
            conv_unit.kc_x = kc_x;

            BlobDesc empty_desc;
            conv_unit.blob = std::make_shared<Blob>(empty_desc);

            conv_units_.emplace_back(conv_unit);
        }
    }

    return TNN_OK;
}

Status ArmDeconvLayerStride::SetSplitBlobDesc(Blob *blob) {
    for (auto &unit : conv_units_) {
        auto desc    = blob->GetBlobDesc();
        desc.dims[1] = unit.param->output_channel;
        desc.dims[2] = desc.dims[2] + unit.kc_y - 1;
        desc.dims[3] = desc.dims[3] + unit.kc_x - 1;

        unit.blob->SetBlobDesc(desc);
    }

    return TNN_OK;
}

Status ArmDeconvLayerStride::SetSplitBlobHandle(Blob *blob, RawBuffer &buf) {
    std::vector<int> blob_data_offset;
    int total_count   = 0;
    int offset        = 0;
    auto data_type    = blob->GetBlobDesc().data_type;
    auto element_size = DataTypeUtils::GetBytesSize(data_type);

    for (auto &unit : conv_units_) {
        auto dims       = unit.blob->GetBlobDesc().dims;
        auto data_count = dims[0] * ROUND_UP(dims[1], 4) * dims[2] * dims[3];
        total_count += data_count;
        blob_data_offset.push_back(offset);
        offset += data_count;
    }

    RawBuffer temp(total_count * element_size);

    for (int i = 0; i < conv_units_.size(); i++) {
        BlobHandle handle;
        handle.base         = temp.force_to<void *>();
        handle.bytes_offset = blob_data_offset[i] * element_size;
        conv_units_[i].blob->SetHandle(handle);
    }

    buf = temp;
    return TNN_OK;
}

Status ArmDeconvLayerStride::CopyOutputSplitBlob(Blob *output) {
    auto param         = reinterpret_cast<ConvLayerParam *>(param_);
    auto dims          = output->GetBlobDesc().dims;
    auto pad_y         = param->pads[2];
    auto pad_x         = param->pads[0];
    auto stride_y      = param->strides[1];
    auto stride_x      = param->strides[0];
    auto batch         = dims[0];
    auto oh            = dims[2];
    auto ow            = dims[3];
    auto output_origin = reinterpret_cast<float *>(GetBlobHandlePtr(output->GetHandle()));

    for (auto &unit : conv_units_) {
        auto stride_dims          = unit.blob->GetBlobDesc().dims;
        auto stride_oh            = stride_dims[2];
        auto stride_ow            = stride_dims[3];
        auto stride_output_origin = reinterpret_cast<float *>(GetBlobHandlePtr(unit.blob->GetHandle()));
        for (int b = 0; b < batch; b++) {
            auto src_b = stride_output_origin + b * ROUND_UP(stride_dims[1], 4) * stride_dims[2] * stride_dims[3];
            auto dst_b = output_origin + b * ROUND_UP(dims[1], 4) * dims[2] * dims[3];
            for (int oz = 0; oz < UP_DIV(dims[1], 4); oz++) {
                auto src_z       = src_b + oz * stride_dims[2] * stride_dims[3] * 4;
                auto dst_z       = dst_b + oz * dims[2] * dims[3] * 4;
                int y_start      = std::ceil(1.0 * (pad_y - unit.y_offset) / stride_y);
                y_start          = std::max(y_start, 0);
                int y_end        = std::floor(1.0 * (pad_y + oh - unit.y_offset - 1) / stride_y);
                y_end            = std::min(y_end, stride_oh);
                auto src_y_start = src_z;
                auto dst_y_start = dst_z + (unit.y_offset - pad_y) * ow * 4;
                for (int y = y_start; y <= y_end; y++) {
                    auto src_y       = src_y_start + y * stride_ow * 4;
                    auto dst_y       = dst_y_start + y * stride_y * ow * 4;
                    int x_start      = std::ceil(1.0 * (pad_x - unit.x_offset) / stride_x);
                    x_start          = std::max(x_start, 0);
                    int x_end        = std::floor(1.0 * (pad_x + ow - unit.x_offset - 1) / stride_x);
                    x_end            = std::min(x_end, stride_ow);
                    auto src_x_start = src_y;
                    auto dst_x_start = dst_y + (unit.x_offset - pad_x) * 4;
                    for (int x = x_start; x <= x_end; x++) {
                        Float4::save(dst_x_start + x * stride_x * 4, Float4::load(src_x_start + x * 4));
                    }
                }
            }
        }
    }

    return TNN_OK;
}

/*
matrix rotate 180
*/
template <typename T>
static inline void _rotete_180(T *ptr, int col, int row) {
    auto rot_ptr = new T[col * row];
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            rot_ptr[(row - i - 1) * col + col - j - 1] = ptr[i * col + j];
        }
    }

    memcpy(ptr, rot_ptr, col * row * sizeof(T));
    delete []rot_ptr;
}

Status ArmDeconvLayerStride::SplitResource() {
    auto param = dynamic_cast<ConvLayerParam *>(param_);
    CHECK_PARAM_NULL(param);
    auto conv_res = dynamic_cast<ConvLayerResource *>(resource_);
    CHECK_PARAM_NULL(conv_res);

    auto sy             = param->strides[1];
    auto sx             = param->strides[0];
    auto ky             = param->kernels[1];
    auto kx             = param->kernels[0];
    auto group          = param->group;
    auto output_channel = param->output_channel;
    auto input_channel  = conv_res->filter_handle.GetDataCount() / group / (kx * ky * output_channel);
    DimsVector res_dims = {input_channel, output_channel, ky, kx};
    for (auto &conv_unit : conv_units_) {
        auto kc_x                      = conv_unit.kc_x;
        auto kc_y                      = conv_unit.kc_y;
        auto x_offset                  = conv_unit.x_offset;
        auto y_offset                  = conv_unit.y_offset;
        auto stride_conv_res           = conv_unit.resource.get();
        DimsVector unit_res_dims       = {output_channel, input_channel, kc_y, kc_x};
        stride_conv_res->filter_handle = RawBuffer(kc_x * kc_y * input_channel * output_channel * sizeof(float));

        for (int ic = 0; ic < input_channel; ic++) {
            for (int oc = 0; oc < output_channel; oc++) {
                auto dst = stride_conv_res->filter_handle.force_to<float *>() +
                           oc * DimsVectorUtils::Count(unit_res_dims, 1) +
                           ic * DimsVectorUtils::Count(unit_res_dims, 2);
                auto src = conv_res->filter_handle.force_to<float *>() + ic * DimsVectorUtils::Count(res_dims, 1) +
                           oc * DimsVectorUtils::Count(res_dims, 2);

                for (int fy = 0; fy < kc_y; fy++) {
                    auto ori_fy = y_offset + fy * sy;
                    for (int fx = 0; fx < kc_x; fx++) {
                        auto ori_fx         = x_offset + fx * sx;
                        dst[fx + fy * kc_x] = src[ori_fy * kx + ori_fx];
                    }
                }

                // Todo: rotate 180
                _rotete_180(dst, kc_x, kc_y);
            }
        }
    }

    return TNN_OK;
}

}  // namespace TNN_NS
