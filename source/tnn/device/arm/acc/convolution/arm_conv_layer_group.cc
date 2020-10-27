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

#include "tnn/device/arm/acc/convolution/arm_conv_layer_group.h"

#include <memory>

#include "tnn/interpreter/raw_buffer.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

Status ArmConvLayerGroup::Init(Context *context, LayerParam *param, LayerResource *resource,
                               const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    Status ret;
    ConvLayerParam *conv_param  = dynamic_cast<ConvLayerParam *>(param);
    ConvLayerResource *conv_res = dynamic_cast<ConvLayerResource *>(resource);
    std::vector<shared_ptr<LayerResource>> resources;

    CHECK_PARAM_NULL(conv_param);
    CHECK_PARAM_NULL(conv_res);

    RETURN_ON_NEQ(ArmLayerAcc::Init(context, param, resource, inputs, outputs), TNN_OK);

    group_ = conv_param->group;

    if (group_inputs_.size() == group_ && group_outputs_.size() == group_) {
        // has been init, just return;
        return TNN_OK;
    } else {
        group_inputs_.clear();
        group_outputs_.clear();
        group_scale_res_.clear();
        conv_acc_impls_.clear();
    }

    for (int g = 0; g < group_; g++) {
        BlobDesc empty_desc;
        group_inputs_.emplace_back(std::make_shared<Blob>(empty_desc));
        group_outputs_.emplace_back(std::make_shared<Blob>(empty_desc));
    }

    RETURN_ON_NEQ(SetGroupParam(group_conv_param_), TNN_OK);
    RETURN_ON_NEQ(SplitResource(resources), TNN_OK);
    RETURN_ON_NEQ(SetSplitBlobDesc(inputs[0], group_inputs_), TNN_OK);
    RETURN_ON_NEQ(SetSplitBlobDesc(outputs[0], group_outputs_), TNN_OK);
    RETURN_ON_NEQ(SetSplitBlobScale(inputs[0], group_inputs_), TNN_OK);
    RETURN_ON_NEQ(SetSplitBlobScale(outputs[0], group_outputs_), TNN_OK);

    for (int g = 0; g < group_; g++) {
        std::vector<Blob *> local_inputs;
        std::vector<Blob *> local_outputs;
        local_inputs.emplace_back(group_inputs_[g].get());
        local_outputs.emplace_back(group_outputs_[g].get());
        std::shared_ptr<ArmLayerAcc> tmp_acc = nullptr;
        if (inputs[0]->GetBlobDesc().data_type == DATA_TYPE_INT8) {
            // CreateImpInt8(local_inputs, local_outputs, group_conv_param_.get(), tmp_acc);
            ArmConvLayerAccFactory::CreateImpInt8(local_inputs, local_outputs, group_conv_param_.get(), tmp_acc);
        } else {
            // CreateImpFP(local_inputs, local_outputs, group_conv_param_.get(), tmp_acc);
            ArmConvLayerAccFactory::CreateImpFP(local_inputs, local_outputs, group_conv_param_.get(), tmp_acc);
        }
        CHECK_PARAM_NULL(tmp_acc);
        RETURN_ON_NEQ(tmp_acc->Init(context_, group_conv_param_.get(), resources[g].get(), local_inputs, local_outputs),
                      TNN_OK);

        conv_acc_impls_.emplace_back(tmp_acc);
    }

    return TNN_OK;
}

ArmConvLayerGroup::~ArmConvLayerGroup() {}

Status ArmConvLayerGroup::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    if (conv_acc_impls_.size() == 0) {
        return Status(TNNERR_LAYER_ERR, "Error: group conv impl is nil");
    } else {
        RETURN_ON_NEQ(SetSplitBlobDesc(inputs[0], group_inputs_), TNN_OK);
        RETURN_ON_NEQ(SetSplitBlobDesc(outputs[0], group_outputs_), TNN_OK);
        for (int g = 0; g < group_; g++) {
            std::vector<Blob *> local_inputs;
            std::vector<Blob *> local_outputs;
            local_inputs.emplace_back(group_inputs_[g].get());
            local_outputs.emplace_back(group_outputs_[g].get());
            RETURN_ON_NEQ(conv_acc_impls_[g]->Reshape(local_inputs, local_outputs), TNN_OK);
        }
    }

    return TNN_OK;
}

Status ArmConvLayerGroup::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    Status ret;
    RawBuffer input_buf;
    RawBuffer output_buf;

    RETURN_ON_NEQ(SetSplitBlobDesc(inputs[0], group_inputs_), TNN_OK);
    RETURN_ON_NEQ(SetSplitBlobDesc(outputs[0], group_outputs_), TNN_OK);
    RETURN_ON_NEQ(SetSplitBlobHandle(group_inputs_, input_buf), TNN_OK);
    RETURN_ON_NEQ(SetSplitBlobHandle(group_outputs_, output_buf), TNN_OK);

    // step 1 : split inputs to group inputs
    CopyInputSplitBlob(inputs[0]);

    // step 2 : group forward
    if (conv_acc_impls_.size()) {
        for (int i = 0; i < conv_acc_impls_.size(); i++) {
            std::vector<Blob *> local_inputs;
            std::vector<Blob *> local_outputs;
            local_inputs.emplace_back(group_inputs_[i].get());
            local_outputs.emplace_back(group_outputs_[i].get());
            CHECK_PARAM_NULL(conv_acc_impls_[i].get());
            RETURN_ON_NEQ(conv_acc_impls_[i]->DoForward(local_inputs, local_outputs), TNN_OK);
        }
    } else {
        return Status(TNNERR_LAYER_ERR, "conv_acc_impl_ is nil");
    }

    // step 3 : merge group outputs into one
    CopyOutputSplitBlob(outputs[0]);

    return TNN_OK;
}

Status ArmConvLayerGroup::SetGroupParam(std::shared_ptr<LayerParam> &group_param) {
    auto conv_param_ = dynamic_cast<ConvLayerParam *>(param_);
    CHECK_PARAM_NULL(conv_param_);

    auto conv_param = new ConvLayerParam();
    CHECK_PARAM_NULL(conv_param);

    *conv_param                = *conv_param_;
    conv_param->output_channel = conv_param->output_channel / conv_param->group;
    conv_param->group          = 1;

    group_param = std::shared_ptr<LayerParam>(conv_param);

    return TNN_OK;
}

Status ArmConvLayerGroup::SetSplitBlobDesc(Blob *blob, std::vector<std::shared_ptr<Blob>> &blobs) {
    auto group_desc    = blob->GetBlobDesc();
    group_desc.dims[1] = group_desc.dims[1] / group_;

    for (int g = 0; g < group_; g++) {
        blobs[g]->SetBlobDesc(group_desc);
    }

    return TNN_OK;
}

Status ArmConvLayerGroup::SetSplitBlobHandle(std::vector<std::shared_ptr<Blob>> &blobs, RawBuffer &buf) {
    auto dims      = blobs[0]->GetBlobDesc().dims;
    auto batch     = dims[0];
    auto data_type = blobs[0]->GetBlobDesc().data_type;

    if (data_type == DATA_TYPE_FLOAT || data_type == DATA_TYPE_BFP16 || data_type == DATA_TYPE_INT8) {
        auto r_split_data_count_per_batch = ROUND_UP(dims[1], 4) * dims[2] * dims[3];
        auto element_size                 = DataTypeUtils::GetBytesSize(data_type);
        RawBuffer temp(group_ * batch * r_split_data_count_per_batch * element_size);

        for (int g = 0; g < group_; g++) {
            BlobHandle handle;
            handle.base = reinterpret_cast<void *>(
                (temp.force_to<char *>() + g * r_split_data_count_per_batch * batch * element_size));
            handle.bytes_offset = 0;
            blobs[g].get()->SetHandle(handle);
        }

        buf = temp;
    } else {
        return Status(TNNERR_LAYER_ERR, "split int8 resource not supported");
    }

    return TNN_OK;
}

Status ArmConvLayerGroup::SetSplitBlobScale(Blob *blob, std::vector<std::shared_ptr<Blob>> &blobs) {
    auto data_type = blob->GetBlobDesc().data_type;

    if (data_type != DATA_TYPE_INT8) {
        // non int8 blob have no scale handle
        return TNN_OK;
    }

    auto int8_blob = reinterpret_cast<BlobInt8 *>(blob);
    auto int8_res  = int8_blob->GetIntResource();

    for (int g = 0; g < group_; g++) {
        auto old_blob = blobs[g];
        auto new_blob = new BlobInt8(old_blob->GetBlobDesc(), old_blob->GetHandle());
        auto new_res  = new IntScaleResource();
        CHECK_PARAM_NULL(new_blob);
        CHECK_PARAM_NULL(new_res);
        auto group_scale_bytes_size = int8_res->scale_handle.GetBytesSize() / group_;
        auto group_bias_bytes_size  = int8_res->bias_handle.GetBytesSize() / group_;

        // set int8 group scale
        if (int8_res->scale_handle.GetDataCount() == 1) {
            new_res->scale_handle = RawBuffer(sizeof(float), int8_res->scale_handle.force_to<char *>());
        } else {
            new_res->scale_handle = RawBuffer(group_scale_bytes_size,
                                              int8_res->scale_handle.force_to<char *>() + g * group_scale_bytes_size);
        }

        // set int8 group bias
        if (int8_res->bias_handle.GetDataCount() == 1) {
            new_res->bias_handle = RawBuffer(sizeof(int32_t), int8_res->bias_handle.force_to<char *>());
        } else {
            new_res->bias_handle =
                RawBuffer(group_bias_bytes_size, int8_res->bias_handle.force_to<char *>() + g * group_bias_bytes_size);
        }

        // set int8 group resource
        new_blob->SetIntResource(new_res);

        // replace blob with int8 blob
        blobs[g] = std::shared_ptr<Blob>(new_blob);
        group_scale_res_.emplace_back(std::shared_ptr<IntScaleResource>(new_res));
    }

    return TNN_OK;
}

template <typename T>
void ArmConvLayerGroup::TransformInput(Blob *input) {
    auto dims       = input->GetBlobDesc().dims;
    auto group_dims = group_inputs_[0]->GetBlobDesc().dims;
    auto batch      = dims[0];

    auto r_split_data_count_per_batch = ROUND_UP(dims[1] / group_, 4) * dims[2] * dims[3];
    auto r_ori_data_count_per_batch   = ROUND_UP(dims[1], 4) * dims[2] * dims[3];

    auto input_origin = reinterpret_cast<T *>(GetBlobHandlePtr(input->GetHandle()));

    for (int b = 0; b < batch; b++) {
        auto input_ptr = input_origin + b * r_ori_data_count_per_batch;
        RawBuffer temp(group_ * r_split_data_count_per_batch * sizeof(T));
        UnpackC4(temp.force_to<T *>(), input_ptr, dims[2] * dims[3], dims[1]);
        for (int g = 0; g < group_; g++) {
            auto group_input_ptr = reinterpret_cast<T *>(GetBlobHandlePtr(group_inputs_[g]->GetHandle()));
            PackC4(group_input_ptr + b * r_split_data_count_per_batch,
                   temp.force_to<T *>() + g * DimsVectorUtils::Count(group_dims, 1, 4),
                   DimsVectorUtils::Count(group_dims, 2, 4), group_dims[1]);
        }
    }
}

static inline void _memcpy_2d(int8_t *dst, int8_t *src, int height, int width, int dst_stride, int src_stride) {
    for (int h = 0; h < height; h++) {
        memcpy(dst + h * dst_stride, src + h * src_stride, width);
    }
}

template <>
void ArmConvLayerGroup::TransformInput<int8_t>(Blob *input) {
    auto input_int8 = reinterpret_cast<BlobInt8 *>(input);

    auto dims       = input_int8->GetBlobDesc().dims;
    auto group_dims = group_inputs_[0]->GetBlobDesc().dims;
    auto batch      = dims[0];

    auto src_stride   = ROUND_UP(dims[1], 4);
    auto dst_stride   = ROUND_UP(group_dims[1], 4);
    auto input_origin = reinterpret_cast<int8_t *>(GetBlobHandlePtr(input_int8->GetHandle()));
    auto plane        = dims[0] * dims[2] * dims[3];

    for (int g = 0; g < group_; g++) {
        auto group_input_ptr = reinterpret_cast<int8_t *>(GetBlobHandlePtr(group_inputs_[g]->GetHandle()));
        _memcpy_2d(group_input_ptr, input_origin + g * group_dims[1], plane, group_dims[1], dst_stride, src_stride);
    }
}

Status ArmConvLayerGroup::CopyInputSplitBlob(Blob *input) {
    auto data_type = input->GetBlobDesc().data_type;

    if (data_type == DATA_TYPE_FLOAT) {
        TransformInput<float>(input);
    } else if (data_type == DATA_TYPE_BFP16) {
        TransformInput<bfp16_t>(input);
    } else if (data_type == DATA_TYPE_INT8) {
        TransformInput<int8_t>(input);
    } else {
        return Status(TNNERR_LAYER_ERR, "split int8 resource not supported");
    }

    return TNN_OK;
}

/*
float and bfp16 layout: NC4HW4
*/
template <typename T>
void ArmConvLayerGroup::TransformOutput(Blob *output) {
    auto dims       = output->GetBlobDesc().dims;
    auto group_dims = group_outputs_[0]->GetBlobDesc().dims;
    auto batch      = dims[0];

    auto r_split_data_count_per_batch = ROUND_UP(dims[1] / group_, 4) * dims[2] * dims[3];
    auto r_ori_data_count_per_batch   = ROUND_UP(dims[1], 4) * dims[2] * dims[3];

    auto output_origin = reinterpret_cast<T *>(GetBlobHandlePtr(output->GetHandle()));

    for (int b = 0; b < batch; b++) {
        auto output_ptr = output_origin + b * r_ori_data_count_per_batch;
        RawBuffer temp(group_ * r_split_data_count_per_batch * sizeof(T));
        for (int g = 0; g < group_; g++) {
            auto group_output_ptr = reinterpret_cast<T *>(GetBlobHandlePtr(group_outputs_[g]->GetHandle()));
            UnpackC4(temp.force_to<T *>() + g * DimsVectorUtils::Count(group_dims, 1, 4),
                     group_output_ptr + b * r_split_data_count_per_batch, DimsVectorUtils::Count(group_dims, 2, 4),
                     group_dims[1]);
        }
        PackC4(output_ptr, temp.force_to<T *>(), dims[2] * dims[3], dims[1]);
    }
}

/*
int8 layout: NHWC4
*/
template <>
void ArmConvLayerGroup::TransformOutput<int8_t>(Blob *output) {
    auto output_int8 = reinterpret_cast<BlobInt8 *>(output);

    auto dims       = output_int8->GetBlobDesc().dims;
    auto group_dims = group_inputs_[0]->GetBlobDesc().dims;
    auto batch      = dims[0];

    auto src_stride    = ROUND_UP(group_dims[1], 4);
    auto dst_stride    = ROUND_UP(dims[1], 4);
    auto output_origin = reinterpret_cast<int8_t *>(GetBlobHandlePtr(output_int8->GetHandle()));
    auto plane         = dims[0] * dims[2] * dims[3];

    for (int g = 0; g < group_; g++) {
        auto group_output_ptr = reinterpret_cast<int8_t *>(GetBlobHandlePtr(group_outputs_[g]->GetHandle()));
        _memcpy_2d(output_origin + g * group_dims[1], group_output_ptr, plane, group_dims[1], dst_stride, src_stride);
    }
}

Status ArmConvLayerGroup::CopyOutputSplitBlob(Blob *output) {
    auto data_type = output->GetBlobDesc().data_type;

    if (data_type == DATA_TYPE_FLOAT) {
        TransformOutput<float>(output);
    } else if (data_type == DATA_TYPE_BFP16) {
        TransformOutput<bfp16_t>(output);
    } else if (data_type == DATA_TYPE_INT8) {
        TransformOutput<int8_t>(output);
    } else {
        return Status(TNNERR_LAYER_ERR, "split int8 resource not supported");
    }

    return TNN_OK;
}

Status ArmConvLayerGroup::SplitResource(std::vector<std::shared_ptr<LayerResource>> &resources) {
    auto conv_param = dynamic_cast<ConvLayerParam *>(param_);
    auto conv_res   = dynamic_cast<ConvLayerResource *>(resource_);

    CHECK_PARAM_NULL(conv_param);
    CHECK_PARAM_NULL(conv_res);

    auto group_filter_bytes_size = conv_res->filter_handle.GetBytesSize() / group_;
    auto origin_filter_ptr       = conv_res->filter_handle.force_to<char *>();

    for (int g = 0; g < group_; g++) {
        auto group_res = new ConvLayerResource();
        // split filter
        group_res->filter_handle = RawBuffer(group_filter_bytes_size, origin_filter_ptr + g * group_filter_bytes_size);

        // split bias if needed
        if (conv_param->bias) {
            auto group_bias_bytes_size = conv_res->bias_handle.GetBytesSize() / group_;
            auto origin_bias_ptr       = conv_res->bias_handle.force_to<char *>();

            group_res->bias_handle = RawBuffer(group_bias_bytes_size, origin_bias_ptr + g * group_bias_bytes_size);
        }

        // split filter scale if int8
        if (conv_res->filter_handle.GetDataType() == DATA_TYPE_INT8) {
            auto scale_handle = conv_res->scale_handle;
            if (scale_handle.GetDataCount() == 1) {
                // channel shared scale
                group_res->scale_handle = RawBuffer(sizeof(float), scale_handle.force_to<char *>());
            } else {
                auto group_scale_bytes_size = scale_handle.GetBytesSize() / group_;
                group_res->scale_handle =
                    RawBuffer(group_scale_bytes_size, scale_handle.force_to<char *>() + g * group_scale_bytes_size);
            }
        }

        resources.emplace_back(std::shared_ptr<LayerResource>(group_res));
    }

    return TNN_OK;
}

}  // namespace TNN_NS
