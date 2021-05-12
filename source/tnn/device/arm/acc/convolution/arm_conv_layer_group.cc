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
#include "tnn/utils/dims_utils.h"

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
        } else if (inputs[0]->GetBlobDesc().data_type == DATA_TYPE_FLOAT ||
                   inputs[0]->GetBlobDesc().data_type == DATA_TYPE_BFP16) {
            // CreateImpFP(local_inputs, local_outputs, group_conv_param_.get(), tmp_acc);
            ArmConvLayerAccFactory::CreateImpFP(local_inputs, local_outputs, group_conv_param_.get(), tmp_acc);
        } 
#if TNN_ARM82
        else if (inputs[0]->GetBlobDesc().data_type == DATA_TYPE_HALF) {
            ArmConvLayerAccFactory::CreateImpHalf(local_inputs, local_outputs, group_conv_param_.get(), tmp_acc);
        }
#endif
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
    if (!conv_acc_impls_.size()) {
        return Status(TNNERR_LAYER_ERR, "conv_acc_impl_ is nil");
    }

    RETURN_ON_NEQ(SetSplitBlobDesc(inputs[0], group_inputs_), TNN_OK);
    RETURN_ON_NEQ(SetSplitBlobDesc(outputs[0], group_outputs_), TNN_OK);    

    auto input_dims    = inputs[0]->GetBlobDesc().dims;
    auto output_dims   = outputs[0]->GetBlobDesc().dims;
    auto data_type     = inputs[0]->GetBlobDesc().data_type;
    auto element_size  = DataTypeUtils::GetBytesSize(data_type);
    auto batch         = input_dims[0];
    auto input_origin  = reinterpret_cast<char *>(GetBlobHandlePtr(inputs[0]->GetHandle()));
    auto output_origin = reinterpret_cast<char *>(GetBlobHandlePtr(outputs[0]->GetHandle()));

    int align_c = 4;
    if (data_type == DATA_TYPE_FLOAT || data_type == DATA_TYPE_BFP16 || data_type == DATA_TYPE_INT8) {
        align_c = 4;
    } else if (data_type == DATA_TYPE_HALF) {
        align_c = 8;
    }

    auto ic_per_group = input_dims[1] / group_;
    auto oc_per_group = output_dims[1] / group_;

    size_t input_step_per_batch        = ROUND_UP(input_dims[1], align_c) * input_dims[2] * input_dims[3];
    size_t input_step_per_group        = ic_per_group * input_dims[2] * input_dims[3];
    size_t input_step_per_group_align  = ROUND_UP(ic_per_group, align_c) * input_dims[2] * input_dims[3];
    size_t output_step_per_batch       = ROUND_UP(output_dims[1], align_c) * output_dims[2] * output_dims[3];
    size_t output_step_per_group       = oc_per_group * output_dims[2] * output_dims[3];
    size_t output_step_per_group_align = ROUND_UP(oc_per_group, align_c) * output_dims[2] * output_dims[3];

    RawBuffer unpack_input_buffer;
    RawBuffer pack_input_buffer;
    RawBuffer unpack_output_buffer;
    RawBuffer pack_output_buffer;
    if (data_type == DATA_TYPE_INT8) {
        RawBuffer unpack_input_data(group_ * input_step_per_group * element_size);
        RawBuffer pack_input_data(group_ * input_step_per_group_align * element_size);
        RawBuffer unpack_output_data(group_ * output_step_per_group * element_size);
        RawBuffer pack_output_data(group_ * output_step_per_group_align * element_size);
        unpack_input_buffer = unpack_input_data;
        pack_input_buffer = pack_input_data;
        unpack_output_buffer = unpack_output_data;
        pack_output_buffer = pack_output_data;
    } else {
        if (ic_per_group % align_c != 0) {
            RawBuffer unpack_data(group_ * input_step_per_group * element_size);
            RawBuffer pack_data(group_ * input_step_per_group_align * element_size);
            unpack_input_buffer = unpack_data;
            pack_input_buffer = pack_data;
        }
        if (oc_per_group % align_c != 0) {
            RawBuffer unpack_data(group_ * output_step_per_group * element_size);
            RawBuffer pack_data(group_ * output_step_per_group_align * element_size);
            unpack_output_buffer = unpack_data;
            pack_output_buffer = pack_data;
        }
    }

    for (int b = 0; b < batch; b++) {
        auto input_batch = input_origin + b * input_step_per_batch * element_size;
        auto output_batch = output_origin + b * output_step_per_batch * element_size;
        auto output_tmp = output_batch;

        if (data_type == DATA_TYPE_INT8) {
            TransformInput(pack_input_buffer.force_to<char *>(),
                        unpack_input_buffer.force_to<char *>(), input_batch,
                        ic_per_group, input_step_per_group, input_step_per_group_align,
                        input_dims, data_type);
            input_batch = pack_input_buffer.force_to<char *>();
            output_tmp = pack_output_buffer.force_to<char *>();
        } else {
            if (ic_per_group % align_c != 0) {
                TransformInput(pack_input_buffer.force_to<char *>(),
                            unpack_input_buffer.force_to<char *>(), input_batch,
                            ic_per_group, input_step_per_group, input_step_per_group_align,
                            input_dims, data_type);
                input_batch = pack_input_buffer.force_to<char *>();
            }

            if (oc_per_group % align_c != 0) {
                output_tmp = pack_output_buffer.force_to<char *>();
            }
        }

        for (int g = 0; g < group_; g++) {
            BlobHandle handle_input;
            BlobHandle handle_output;
            handle_input.base = reinterpret_cast<void *>((input_batch + g * input_step_per_group_align * element_size));
            handle_input.bytes_offset = 0;
            group_inputs_[g].get()->SetHandle(handle_input);
            handle_output.base = reinterpret_cast<void *>((output_tmp + g * output_step_per_group_align * element_size));
            handle_output.bytes_offset = 0;
            group_outputs_[g].get()->SetHandle(handle_output);

            std::vector<Blob *> local_inputs;
            std::vector<Blob *> local_outputs;
            local_inputs.emplace_back(group_inputs_[g].get());
            local_outputs.emplace_back(group_outputs_[g].get());
            CHECK_PARAM_NULL(conv_acc_impls_[g].get());
            RETURN_ON_NEQ(conv_acc_impls_[g]->DoForward(local_inputs, local_outputs), TNN_OK);
        }

        if (data_type == DATA_TYPE_INT8) {
            TransformOutput(pack_output_buffer.force_to<char *>(),
                            unpack_output_buffer.force_to<char *>(), output_batch,
                            oc_per_group, output_step_per_group, output_step_per_group_align,
                            output_dims, data_type);
        } else {
            if (oc_per_group % align_c != 0) {
                TransformOutput(pack_output_buffer.force_to<char *>(),
                                unpack_output_buffer.force_to<char *>(), output_batch,
                                oc_per_group, output_step_per_group, output_step_per_group_align,
                                output_dims, data_type);
            }
        }
    }
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
    group_desc.dims[0] = 1;
    group_desc.dims[1] = group_desc.dims[1] / group_;

    for (int g = 0; g < group_; g++) {
        blobs[g]->SetBlobDesc(group_desc);
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

static inline void _memcpy_2d(int8_t *dst, int8_t *src, int height, int width, int dst_stride, int src_stride) {
    for (int h = 0; h < height; h++) {
        memcpy(dst + h * dst_stride, src + h * src_stride, width);
    }
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

void ArmConvLayerGroup::TransformInput(char *packed, char *unpacked, char *src, 
                    size_t ic_per_group, size_t group_step, size_t group_step_align,
                    DimsVector dims, DataType data_type) {
    if (data_type == DATA_TYPE_FLOAT) {
        // FLOAT NC4HW4
        float *packed_data = reinterpret_cast<float *>(packed);
        float *unpacked_data = reinterpret_cast<float *>(unpacked);
        float *src_data = reinterpret_cast<float *>(src);
        UnpackC4(unpacked_data, src_data, dims[2] * dims[3], dims[1]);
        for (int g = 0; g < group_; g++) {
            PackC4(packed_data + g * group_step_align, 
                   unpacked_data + g * group_step, 
                   dims[2] * dims[3], ic_per_group);
        }
    } else if (data_type == DATA_TYPE_BFP16) {
        // BFP16 NC4HW4
        bfp16_t *packed_data = reinterpret_cast<bfp16_t *>(packed);
        bfp16_t *unpacked_data = reinterpret_cast<bfp16_t *>(unpacked);
        bfp16_t *src_data = reinterpret_cast<bfp16_t *>(src);
        UnpackC4(unpacked_data, src_data, dims[2] * dims[3], dims[1]);
        for (int g = 0; g < group_; g++) {
            PackC4(packed_data + g * group_step_align, 
                   unpacked_data + g * group_step, 
                   dims[2] * dims[3], ic_per_group);
        }
    } else if (data_type == DATA_TYPE_INT8) {
        // INT8 NHWC
        int8_t *packed_data = reinterpret_cast<int8_t *>(packed);
        int8_t *src_data = reinterpret_cast<int8_t *>(src);
        size_t ic_per_group_align = ROUND_UP(ic_per_group, 4);
        size_t ic_align = ROUND_UP(dims[1], 4);
        for (int g = 0; g < group_; g++) {
            _memcpy_2d(packed_data + g * group_step_align, 
                       src_data + g * ic_per_group, dims[2] * dims[3],
                       ic_per_group, ic_per_group_align, ic_align);
        }
    }
#if TNN_ARM82
    else if (data_type == DATA_TYPE_HALF) {
        // FP16 NC8HW8
        fp16_t *packed_data = reinterpret_cast<fp16_t *>(packed);
        fp16_t *unpacked_data = reinterpret_cast<fp16_t *>(unpacked);
        fp16_t *src_data = reinterpret_cast<fp16_t *>(src);
        UnpackC8(unpacked_data, src_data, dims[2] * dims[3], dims[1]);
        for (int g = 0; g < group_; g++) {
            PackC8(packed_data + g * group_step_align, 
                   unpacked_data + g * group_step, 
                   dims[2] * dims[3], ic_per_group);
        }
    }
#endif
    return;
}

void ArmConvLayerGroup::TransformOutput(char *packed, char *unpacked, char *dst, 
                    size_t oc_per_group, size_t group_step, size_t group_step_align,
                    DimsVector dims, DataType data_type) {
    if (data_type == DATA_TYPE_FLOAT) {
        // FLOAT NC4HW4
        float *packed_data = reinterpret_cast<float *>(packed);
        float *unpacked_data = reinterpret_cast<float *>(unpacked);
        float *dst_data = reinterpret_cast<float *>(dst);
        for (int g = 0; g < group_; g++) {
            UnpackC4(unpacked_data + g * group_step, 
                     packed_data + g * group_step_align, 
                     dims[2] * dims[3], oc_per_group);
        }
        PackC4(dst_data, unpacked_data, dims[2] * dims[3], dims[1]);
    } else if (data_type == DATA_TYPE_BFP16) {
        // BFP16 NC4HW4
        bfp16_t *packed_data = reinterpret_cast<bfp16_t *>(packed);
        bfp16_t *unpacked_data = reinterpret_cast<bfp16_t *>(unpacked);
        bfp16_t *dst_data = reinterpret_cast<bfp16_t *>(dst);
        for (int g = 0; g < group_; g++) {
            UnpackC4(unpacked_data + g * group_step, 
                     packed_data + g * group_step_align, 
                     dims[2] * dims[3], oc_per_group);
        }
        PackC4(dst_data, unpacked_data, dims[2] * dims[3], dims[1]);
    } else if (data_type == DATA_TYPE_INT8) {
        // INT8 NHWC
        int8_t *packed_data = reinterpret_cast<int8_t *>(packed);
        int8_t *dst_data = reinterpret_cast<int8_t *>(dst);
        size_t oc_per_group_align = ROUND_UP(oc_per_group, 4);
        size_t oc_align = ROUND_UP(dims[1], 4);
        for (int g = 0; g < group_; g++) {
            _memcpy_2d(dst_data + g * oc_per_group, 
                       packed_data + g * group_step_align,
                       dims[2] * dims[3], oc_per_group, 
                       oc_align, oc_per_group_align);
        }
    }
#if TNN_ARM82
    else if (data_type == DATA_TYPE_HALF) {
        // FP16 NC8HW8
        fp16_t *packed_data = reinterpret_cast<fp16_t *>(packed);
        fp16_t *unpacked_data = reinterpret_cast<fp16_t *>(unpacked);
        fp16_t *dst_data = reinterpret_cast<fp16_t *>(dst);
        for (int g = 0; g < group_; g++) {
            UnpackC8(unpacked_data + g * group_step,
                     packed_data + g * group_step_align,
                     dims[2] * dims[3], oc_per_group);
        }
        PackC8(dst_data, unpacked_data, dims[2] * dims[3], dims[1]);
    }
#endif
    return;
}

}  // namespace TNN_NS
