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

#include <algorithm>
#include <cmath>

#include "tnn/device/arm/acc/arm_layer_acc.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

DECLARE_ARM_ACC(StrideSlice, LAYER_STRIDED_SLICE);

static Status ExecStrideSlice(Blob *input_blob, Blob *output_blob, const std::vector<int> &begins,
                              const std::vector<int> &ends, const std::vector<int> &strides) {
    auto dims_input  = input_blob->GetBlobDesc().dims;
    auto dims_output = output_blob->GetBlobDesc().dims;
    int input_slice  = UP_DIV(dims_input[1], 4);
    int output_slice = UP_DIV(dims_output[1], 4);

    // support maximum dim 5
    int input_strides[4];
    int output_strides[4];
    input_strides[0]  = DimsVectorUtils::Count(dims_input, 2) * 4 * input_slice;
    output_strides[0] = DimsVectorUtils::Count(dims_output, 2) * 4 * output_slice;
    for (int i = 1; i < 4; i++) {
        input_strides[i]  = DimsVectorUtils::Count(dims_input, i + 1) * 4;
        output_strides[i] = DimsVectorUtils::Count(dims_output, i + 1) * 4;
    }

    if (output_blob->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        float *input_data  = reinterpret_cast<float *>(GetBlobHandlePtr(input_blob->GetHandle()));
        float *output_data = reinterpret_cast<float *>(GetBlobHandlePtr(output_blob->GetHandle()));

        int nn = 0, nc = 0, nh = 0, nw = 0, nx = 0;
        if (begins.size() == 5) {
            for (int n = begins[0]; n < ends[0]; n += strides[0], nn++) {
                auto input_n  = input_data + n * input_strides[0];
                auto output_n = output_data + nn * output_strides[0];
                nc            = 0;
                for (int c = begins[1]; c < ends[1]; c += strides[1], nc++) {
                    auto zi = c / 4, ri = c % 4;
                    auto zo = nc / 4, ro = nc % 4;

                    auto input_c  = input_n + zi * input_strides[1];
                    auto output_c = output_n + zo * output_strides[1];
                    nh            = 0;
                    for (int h = begins[2]; h < ends[2]; h += strides[2], nh++) {
                        auto input_h  = input_c + h * input_strides[2];
                        auto output_h = output_c + nh * output_strides[2];
                        nw            = 0;
                        for (int w = begins[3]; w < ends[3]; w += strides[3], nw++) {
                            auto input_w  = input_h + w * input_strides[3];
                            auto output_w = output_h + nw * output_strides[3];
                            nx            = 0;
                            for (int x = begins[4]; x < ends[4]; x += strides[4], nx++) {
                                output_w[nx * 4 + ro] = input_w[x * 4 + ri];
                            }
                        }
                    }
                }
            }
        } else if (begins.size() == 4) {
            for (int n = begins[0]; n < ends[0]; n += strides[0], nn++) {
                auto input_n  = input_data + n * input_strides[0];
                auto output_n = output_data + nn * output_strides[0];
                nc            = 0;
                for (int c = begins[1]; c < ends[1]; c += strides[1], nc++) {
                    auto zi = c / 4, ri = c % 4;
                    auto zo = nc / 4, ro = nc % 4;

                    auto input_c  = input_n + zi * input_strides[1];
                    auto output_c = output_n + zo * output_strides[1];
                    nh            = 0;
                    for (int h = begins[2]; h < ends[2]; h += strides[2], nh++) {
                        auto input_h  = input_c + h * input_strides[2];
                        auto output_h = output_c + nh * output_strides[2];
                        nw            = 0;
                        for (int w = begins[3]; w < ends[3]; w += strides[3], nw++) {
                            output_h[nw * 4 + ro] = input_h[w * 4 + ri];
                        }
                    }
                }
            }
        } else if (begins.size() == 3) {
            for (int n = begins[0]; n < ends[0]; n += strides[0], nn++) {
                auto input_n  = input_data + n * input_strides[0];
                auto output_n = output_data + nn * output_strides[0];
                nc            = 0;
                for (int c = begins[1]; c < ends[1]; c += strides[1], nc++) {
                    auto zi = c / 4, ri = c % 4;
                    auto zo = nc / 4, ro = nc % 4;

                    auto input_c  = input_n + zi * input_strides[1];
                    auto output_c = output_n + zo * output_strides[1];
                    nh            = 0;
                    for (int h = begins[2]; h < ends[2]; h += strides[2], nh++) {
                        output_c[nh * 4 + ro] = input_c[h * 4 + ri];
                    }
                }
            }
        } else if (begins.size() == 2) {
            for (int n = begins[0]; n < ends[0]; n += strides[0], nn++) {
                auto input_n  = input_data + n * input_strides[0];
                auto output_n = output_data + nn * output_strides[0];
                nc            = 0;
                for (int c = begins[1]; c < ends[1]; c += strides[1], nc++) {
                    output_n[nc] = input_n[c];
                }
            }
        }
    } else {
        return Status(TNNERR_LAYER_ERR, "NO IMPLEMENT FOR int8/bfp16 StrideSlice, in todo list");
    }
    return TNN_OK;
}

Status ArmStrideSliceLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<StrideSliceLayerParam *>(param_);
    if (!layer_param) {
        LOGE("Error: StrideSliceLayerParam is nil\n");
        return Status(TNNERR_MODEL_ERR, "Error: StrideSliceLayerParam is nil");
    }

    Blob *input_blob  = inputs[0];
    Blob *output_blob = outputs[0];
    auto dims_input   = input_blob->GetBlobDesc().dims;
    auto dims_output  = output_blob->GetBlobDesc().dims;
    auto dim_size     = dims_output.size();
    if ((dim_size > 5 || dim_size < 2) || dim_size != dims_input.size()) {
        return Status(TNNERR_MODEL_ERR, "Error: StrideSliceLayerParam not support!");
    }

    auto begins  = layer_param->begins;
    auto ends    = layer_param->ends;
    auto strides = layer_param->strides;
    std::reverse(begins.begin(), begins.end());
    std::reverse(ends.begin(), ends.end());
    std::reverse(strides.begin(), strides.end());

    for (int i = 0; i < ends.size(); ++i) {
        if (ends[i] == 0) {
            ends[i] = input_blob->GetBlobDesc().dims[i];
        }
    }

    return ExecStrideSlice(input_blob, output_blob, begins, ends, strides);
}

REGISTER_ARM_ACC(StrideSlice, LAYER_STRIDED_SLICE)
REGISTER_ARM_LAYOUT(LAYER_STRIDED_SLICE, DATA_FORMAT_NC4HW4)

DECLARE_ARM_ACC(StrideSliceV2, LAYER_STRIDED_SLICE_V2);

static Status FastSliceForHW(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs,
                             StrideSliceV2LayerParam *param) {
    Blob *input_blob        = inputs[0];
    Blob *output_blob       = outputs[0];
    auto dims_input         = input_blob->GetBlobDesc().dims;
    auto dims_output        = output_blob->GetBlobDesc().dims;
    auto begin              = param->begins[0];
    auto end                = param->ends[0];
    auto axis               = param->axes[0];
    auto *input_ptr         = reinterpret_cast<char *>(GetBlobHandlePtr(input_blob->GetHandle()));
    auto *output_ptr        = reinterpret_cast<char *>(GetBlobHandlePtr(output_blob->GetHandle()));
    const int batch         = DimsFunctionUtils::GetDim(dims_output, 0);
    const int channel       = DimsFunctionUtils::GetDim(dims_output, 1);
    const int count         = DimsVectorUtils::Count(dims_output, 2, axis);
    const int original_axis = DimsFunctionUtils::GetDim(dims_input, axis);
    const int slice_axis    = DimsFunctionUtils::GetDim(dims_output, axis);
    const int step          = DimsVectorUtils::Count(dims_output, axis + 1);
    int channel_stride      = 0;
    int byte_size           = 0;
    const auto data_type    = output_blob->GetBlobDesc().data_type;
    if (data_type == DATA_TYPE_FLOAT) {
        channel_stride = 4;
        byte_size      = sizeof(float);
    } else if (data_type == DATA_TYPE_HALF) {
        channel_stride = 8;
        byte_size      = sizeof(fp16_t);

    } else {
        LOGE("ArmStrideSliceV2LayerAcc does not support data type: %d", data_type);
        return {TNNERR_UNSUPPORT_NET, "ArmStrideSliceV2LayerAcc does not support data type\n"};
    }
    int channel_up = UP_DIV(channel, channel_stride);
    // end may be max int value;
    int step_size = step * channel_stride * byte_size;
    for (int b = 0; b < batch; ++b) {
        auto *input_batch_ptr  = input_ptr + b * channel_up * count * original_axis * step_size;
        auto *output_batch_ptr = output_ptr + b * channel_up * count * slice_axis * step_size;
        for (int c = 0; c < channel_up * count; ++c) {
            auto *input_slice  = input_batch_ptr + (c * original_axis + begin) * step_size;
            auto *output_slice = output_batch_ptr + (c * slice_axis) * step_size;
            memcpy(output_slice, input_slice, (end - begin) * step_size);
        }
    }
    return TNN_OK;
}

Status ArmStrideSliceV2LayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<StrideSliceV2LayerParam *>(param_);
    if (!layer_param) {
        LOGE("Error: StrideSliceV2LayerParam is nil\n");
        return Status(TNNERR_MODEL_ERR, "Error: StrideSliceV2LayerParam is nil");
    }

    Blob *input_blob  = inputs[0];
    Blob *output_blob = outputs[0];
    auto dims_input   = input_blob->GetBlobDesc().dims;
    auto dims_output  = output_blob->GetBlobDesc().dims;
    auto dim_size     = dims_output.size();
    if ((dim_size > 5 || dim_size < 2) || dim_size != dims_input.size()) {
        return Status(TNNERR_MODEL_ERR, "Error: StrideSliceV2LayerParam not support!");
    }

    auto begins  = layer_param->begins;
    auto ends    = layer_param->ends;
    auto strides = layer_param->strides;
    auto axes    = layer_param->axes;
    // optimize case: axes.size = 1
    if (axes.size() == 1 && (axes[0] > 1) && strides[0] == 1) {
        return FastSliceForHW(inputs, outputs, layer_param);
    }

    std::vector<int> rectified_begins(dim_size, 0);
    std::vector<int> rectified_ends(dim_size, 0);
    std::vector<int> rectified_strides(dim_size, 0);
    for (int i = 0, axes_idx = 0; i < dim_size; ++i) {
        if (axes_idx >= axes.size() || i != axes[axes_idx]) {
            rectified_begins[i]  = 0;
            rectified_ends[i]    = dims_input[i];
            rectified_strides[i] = 1;
        } else {
            rectified_begins[i]  = begins[axes_idx];
            rectified_ends[i]    = ends[axes_idx];
            rectified_strides[i] = strides[axes_idx];
            axes_idx += 1;
        }
    }

    return ExecStrideSlice(input_blob, output_blob, rectified_begins, rectified_ends, rectified_strides);
}

REGISTER_ARM_ACC(StrideSliceV2, LAYER_STRIDED_SLICE_V2)
REGISTER_ARM_LAYOUT(LAYER_STRIDED_SLICE_V2, DATA_FORMAT_NC4HW4)

}  // namespace TNN_NS
