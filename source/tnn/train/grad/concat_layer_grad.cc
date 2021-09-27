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

// author: sanerzheng@tencent.com

#include "tnn/train/grad/layer_grad.h"
#include "tnn/train/operations/op_builder.h"
// TODO: 去掉arm依赖，改成设备无关的，否则后续加入其他设备支持时，编译不过
#include "tnn/device/arm/arm_util.h"

namespace TNN_NS {
namespace train {
DECLARE_LAYER_GRAD(Concat, LAYER_CONCAT);
bool CheckConcatShape(DimsVector shape1, DimsVector shape2, int exclude_axis) {
    if (shape1.size() != shape2.size()) {
        LOGE("shape1 dim size %d  shape2 dim size %d\n", (int)shape1.size(), (int)shape2.size());
        return false;
    }

    int i = 0;
    for (; i < shape1.size(); i++) {
        if ((i != exclude_axis && shape1[i] != shape2[i]) || (shape1[i] <= 0 || shape2[i] <= 0)) {
            LOGE("dim[%d] not match (shape1:%d, shape2:%d)\n", i, shape1[i], shape2[i]);
            return false;
        }
    }

    if (exclude_axis >= shape1.size()) {
        LOGE("exclude_axis:%d out of shape size:%d\n", exclude_axis, (int)shape1.size());
        return false;
    }
    return true;
}
DimsVector GetCXRoundDims(DimsVector &dims, const int round) {
    DimsVector round_dims = {dims[0], UP_DIV(dims[1], round)};
    for (int i = 2; i < dims.size(); ++i) {
        round_dims.push_back(dims[i]);
    }
    round_dims.push_back(round);
    return round_dims;
}
inline char *GetBlobPtr(BlobHandle handle) {
    return static_cast<char *>(handle.base) + handle.bytes_offset;
}
template <typename T>
int split(RawBuffer *output_grad, std::vector<std::shared_ptr<RawBuffer>> &inputs_grad, int axis) {
    auto output_dims             = output_grad->GetBufferDims();
    auto output_format           = output_grad->GetDataFormat(); // nc4hw4 or nchw
    DimsVector round_output_dims = output_format == DATA_FORMAT_NC4HW4 ? GetCXRoundDims(output_dims, 4) : output_dims;
    auto slice_count             = DimsVectorUtils::Count(round_output_dims, 0, axis);
    auto output_stride           = DimsVectorUtils::Count(round_output_dims, axis);
    auto *output_origin          = output_grad->force_to<T *>();

    for (int n = 0; n < slice_count; n++) {
        auto output_ptr = output_origin + n * output_stride;
        for (int b = 0; b < inputs_grad.size(); b++) {
            auto input      = inputs_grad[b];
            auto input_dims = input->GetBufferDims();
            DimsVector round_input_dims =
                input->GetDataFormat() == DATA_FORMAT_NC4HW4 ? GetCXRoundDims(input_dims, 4) : input_dims;
            ;
            auto input_stride = DimsVectorUtils::Count(round_input_dims, axis);
            auto input_ptr    = input.get()->force_to<T *>() + n * input_stride;
            memcpy(input_ptr, output_ptr, input_stride * sizeof(T));
            output_ptr += input_stride;
        }
    }

    return 0;
}

// only use in nc4hw4 and axis is 1
template <typename T> int split_channel(RawBuffer *output_grad, std::vector<std::shared_ptr<RawBuffer>> &inputs_grad) {
    auto dims_output    = output_grad->GetBufferDims();
    auto output_stride  = DimsVectorUtils::Count(dims_output, 2) * ROUND_UP(dims_output[1], 4);
    auto *output_origin = output_grad->force_to<T *>();
    RawBuffer buffer(dims_output[1] * DimsVectorUtils::Count(dims_output, 2));
    T *unpack_buf = buffer.force_to<T *>();
    int area      = DimsVectorUtils::Count(dims_output, 2);
    for (int n = 0; n < dims_output[0]; n++) {
        auto *output_ptr = output_origin + n * output_stride;
        auto *unpack_ptr = unpack_buf;
        UnpackC4(unpack_buf, output_ptr, area, dims_output[1]);
        for (int b = 0; b < inputs_grad.size(); b++) {
            auto input      = inputs_grad[b];
            auto dims_input = input->GetBufferDims();
            auto c_r4       = ROUND_UP(dims_input[1], 4);
            auto input_ptr  = input.get()->force_to<T *>() + n * c_r4 * area;
            PackC4(input_ptr, unpack_ptr, area, dims_input[1]);
            unpack_ptr += dims_input[1] * area;
        }
    }

    return 0;
}

Status ConcatLayerGrad::OnGrad(const BaseLayer *layer, TrainContext &context) {
    auto inputs  = layer->input_blobs_;
    auto outputs = layer->output_blobs_;
    if (inputs.size() <= 0 || outputs.size() != 1) {
        return Status(TNN_TRAIN_ERROR, "input size or output size not match in ConcatLayerGrad");
    }
    auto input0_desc      = inputs[0]->GetBlobDesc();
    auto output_desc      = outputs[0]->GetBlobDesc();
    auto input0_data_type = input0_desc.data_type;
    auto output_data_type = output_desc.data_type;
    auto input0_dims      = input0_desc.dims;
    auto output_dims      = output_desc.dims;
    if (input0_data_type != output_data_type) {
        return Status(TNN_TRAIN_ERROR, "input datatype and output datatype not match in ConcatLayerGrad");
    }
    if ((input0_desc.data_format != DATA_FORMAT_NC4HW4 && input0_desc.data_format != DATA_FORMAT_NCHW) ||
        input0_desc.data_format != output_desc.data_format) {
        return Status(TNN_TRAIN_ERROR, "output dataformat not match in ConcatLayerGrad");
    }
    if (output_data_type != DATA_TYPE_BFP16 && output_data_type != DATA_TYPE_FLOAT) {
        return Status(TNN_TRAIN_ERROR, "output datatype not match in ConcatLayerGrad");
    }
    auto layer_param = dynamic_cast<SoftmaxLayerParam *>(layer->param_);
    if (layer_param == nullptr)
        return Status(TNN_TRAIN_ERROR, "ConcatLayer param axis error");
    auto iter_output_grad = context.backward_grads_blob.find(outputs[0]);
    if (iter_output_grad == context.backward_grads_blob.end()) {
        return Status(TNN_TRAIN_ERROR, "ConcatLayerGrad output grad not found");
    }
    int axis = layer_param->axis;
    if (axis < 0) {
        axis += (int)input0_dims.size();
        layer_param->axis = axis;
    }
    if (axis < 0 || axis > input0_dims.size()) {
        LOGE("ConcatLayerGrad Error:  axis(%d) is invalid\n", axis);
        return Status(TNN_TRAIN_ERROR, "ConcatLaConcatLayerGradyer Error: axis is invalid");
    }

    size_t i                = 0;
    auto last_shape         = inputs[i]->GetBlobDesc().dims;
    int out_concat_dim_size = 0;
    for (; i < inputs.size(); i++) {
        auto input_blob = inputs[i];
        auto cur_shape  = inputs[i]->GetBlobDesc().dims;
        if (!CheckConcatShape(last_shape, cur_shape, axis)) {
            LOGE("ConcatLayerGrad Error:inputs dims not match in %d dim", axis);
            return Status(TNN_TRAIN_ERROR, "ConcatLayer's inputs can not be concatenated");
        }
    }
    std::vector<std::shared_ptr<RawBuffer>> inputs_grad;
    inputs_grad.resize(inputs.size());
    for (int i = 0; i < inputs.size(); ++i) {
        auto input_desc = inputs[i]->GetBlobDesc();
        int bytes_size  = CalculateElementCount(input_desc) * DataTypeUtils::GetBytesSize(input_desc.data_type);
        inputs_grad[i]  = std::make_shared<RawBuffer>(bytes_size, input_desc.dims);
        inputs_grad[i]->SetDataFormat(input_desc.data_format);
        inputs_grad[i]->SetDataType(input_desc.data_type);
    }
    bool concat_c4 = true;
    for (int i = 0; i < inputs.size() - 1; i++) {
        auto dims = inputs[i]->GetBlobDesc().dims;
        if (dims.size() <= 1 || dims[1] % 4 != 0) {
            concat_c4 = false;
            break;
        }
    }
    if (input0_desc.data_format == DATA_FORMAT_NC4HW4 && axis == 1 && !concat_c4) {
        switch (output_data_type) {
        case DATA_TYPE_FLOAT:
            split_channel<float>(iter_output_grad->second.get(), inputs_grad);
            break;
        case DATA_TYPE_BFP16:
            split_channel<bfp16_t>(iter_output_grad->second.get(), inputs_grad);
            break;
        default:
            return Status(TNN_TRAIN_ERROR, "ConcatLayerGrad not support datatype error");
        }
    } else {
        switch (output_data_type) {
        case DATA_TYPE_FLOAT:
            split<float>(iter_output_grad->second.get(), inputs_grad, axis);
            break;
        case DATA_TYPE_BFP16:
            split<bfp16_t>(iter_output_grad->second.get(), inputs_grad, axis);
            break;
        default:
            return Status(TNN_TRAIN_ERROR, "Calcute ConcatLayerGrad error");
        }
    }

    for (int i = 0; i < inputs_grad.size(); ++i) {
        UpdateGradValue(inputs[i], inputs_grad[i], context);
    }
    return Status(TNN_OK);
}
REGISTER_LAYER_GRAD(Concat, LAYER_CONCAT);

} // namespace train
} // namespace TNN_NS