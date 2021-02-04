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

#include "tnn/utils/dims_vector_utils.h"

#include <cmath>
#include <climits>

namespace TNN_NS {

int DimsVectorUtils::Count(DimsVector dims, int start_index, int end_index) {
    if (dims.size() < start_index) {
        return 0;
    }

    if (-1 == end_index || end_index > dims.size()) {
        end_index = static_cast<int>(dims.size());
    }

    int result = 1;
    for (int index = start_index; index < end_index; ++index) {
        result *= dims[index];
    }
    return result;
}

DimsVector DimsVectorUtils::Max(DimsVector dims0, DimsVector dims1, int start_index, int end_index) {
    DimsVector max_dims;
    DimsVector small_dims;
    if (dims0.size() >= dims1.size()) {
        max_dims   = dims0;
        small_dims = dims1;
    } else {
        max_dims   = dims1;
        small_dims = dims0;
    }

    if (small_dims.size() <= start_index) {
        return max_dims;
    }

    if (-1 == end_index || end_index > small_dims.size()) {
        end_index = static_cast<int>(small_dims.size());
    }

    for (int i = start_index; i < end_index; i++) {
        max_dims[i] = std::max(max_dims[i], small_dims[i]);
    }

    return max_dims;
}

bool DimsVectorUtils::Equal(DimsVector dims0, DimsVector dims1, int start_index, int end_index) {
    if (dims0.size() <= start_index) {
        return false;
    }

    if (-1 == end_index || end_index > dims0.size()) {
        end_index = static_cast<int>(dims0.size());
    }

    if (dims0.size() != dims1.size()) {
        return false;
    }

    for (int i = start_index; i < end_index; i++) {
        if (dims0[i] != dims1[i]) {
            return false;
        }
    }
    return true;
}

DimsVector DimsVectorUtils::Expand(DimsVector dims0, DimsVector dims1, Status *status) {
    DimsVector max_dims;
    DimsVector min_dims;
    if (dims0.size() >= dims1.size()) {
        max_dims   = dims0;
        min_dims = dims1;
    } else {
        max_dims   = dims1;
        min_dims = dims0;
    }
    
    auto output_dims = max_dims;
    const int offset = (int)(max_dims.size() - min_dims.size());
    for(int i = 0; i < min_dims.size(); ++i) {
        if(max_dims[offset + i] == 1 || max_dims[offset + i] == -1) {
            if (min_dims[i] > output_dims[offset + i]) {
                output_dims[offset + i] = min_dims[i];
            }
        } else if (max_dims[offset + i] != min_dims[i]) {
            if (status) {
                *status = Status(TNNERR_PARAM_ERR, "expand param dims error");
            }
        }
    }

    return output_dims;
}

DimsVector DimsVectorUtils::Upsample(const DimsVector input_dims,
                                     std::vector<float> scales, std::vector<int> sizes, int mode, Status *status) {
    int num          = input_dims[0];
    int channels   = input_dims[1];
    int height       = input_dims[2];
    int width        = input_dims[3];
    
    int width_out    = 0;
    int height_out   = 0;
    
    if (sizes.size() <= 0) {
        if (mode == 1 || mode == 2 || mode == 3) {
            //floor is wrong for some model
            width_out  = int(round(width * scales[0]));
            height_out = int(round(height * scales[1]));
        } else {
            if (status) {
                *status = Status(TNNERR_PARAM_ERR, "unsupport upsample type");
            }
            return DimsVector();
        }
    } else {
        width_out  = sizes[0];
        height_out = sizes[1];
    }

    if (width_out <= 0 || height_out <= 0) {
        if (status) {
            *status = Status(TNNERR_PARAM_ERR, "UpsampleLayer has invalid output shape");
        }
    }
    
    return {num, channels, height_out, width_out};
}

DimsVector DimsVectorUtils::Range(const RangeData start, const RangeData limit,
                        const RangeData delta, DataType type, Status *status) {
    int count = 0;
    if (type == DATA_TYPE_FLOAT) {
        count = ceil((limit.f - start.f) / delta.f);
    } else if (type == DATA_TYPE_INT32) {
        count = ceil((limit.i - start.i) / delta.i);
    } else {
        if (status) {
            *status = Status(TNNERR_PARAM_ERR, "RangeLayer has invalid type");
        }
    }
    
    count = count >= 0 ? count : 0;
    
    return {count};
}

DimsVector DimsVectorUtils::Reshape(const DimsVector input_dims, const DimsVector shape,
                                    const int axis, const int num_axes, Status *status) {

    int output_size = shape.size() + axis;
    DimsVector output_dims(output_size, 1);

    for(int i = 0; i < axis; ++i) {
        output_dims[i] = input_dims[i];
    }

    int infer_dim_count = 0;
    int infer_dim_pos   = -1;
    for (int i = axis, j = 0; i < num_axes; i++, j++) {
        if (shape[j] == -1) {
            infer_dim_count += 1;
            infer_dim_pos  = i;
            output_dims[i] = 1;
        } else if (shape[j] == 0) {
            output_dims[i] = input_dims[i];
        } else {
            output_dims[i] = shape[j];
        }
    }
    
    // temporary fix reshpae init error
    if (infer_dim_count == 0 && infer_dim_pos == -1) {
        return output_dims;
    }

    if (infer_dim_count != 1 || infer_dim_pos == -1) {
        if (status) {
            *status = Status(TNNERR_PARAM_ERR, "reshape param size error");
        }
        return DimsVector();
    }

    int in_cnt  = DimsVectorUtils::Count(input_dims);
    int out_cnt = DimsVectorUtils::Count(output_dims);
    if (0 == out_cnt) {
        if (status) {
            *status = Status(TNNERR_COMMON_ERROR, "Error: blob count is zero");
        }
    }
    
    int infer_dim_v = in_cnt / out_cnt;
    if (infer_dim_v <= 0) {
        if (status) {
            *status = Status(TNNERR_COMMON_ERROR, "Error: blob shape is zero");
        }
    }
    output_dims[infer_dim_pos] = infer_dim_v;
    return output_dims;
}

DimsVector DimsVectorUtils::StrideSlice(const DimsVector input_dims,
                                        DimsVector& begins, DimsVector& ends, const DimsVector strides,
                                        const DimsVector axes, Status *status) {
    if (axes.size() != begins.size() || axes.size() != ends.size() || axes.size() != strides.size()) {
        if (status) {
            *status = Status(TNNERR_PARAM_ERR, "StrideSliceV2Layer param of axes, ends, strides size is invalid");
            return DimsVector();
        }
    }

    auto output_dims = input_dims;

    //前闭后开区间
    for (int i = 0; i < axes.size(); i++) {
        int index = axes[i];
        if (begins[i] < 0) {
            begins[i] += input_dims[index];
        }

        if (ends[i] == INT_MAX) {
            ends[i] = input_dims[index];
        }

        if (ends[i] < 0) {
            ends[i] += input_dims[index];
        }

        if (begins[i] >= ends[i]) {
            if (status) {
                *status = Status(TNNERR_PARAM_ERR, "StrideSliceV2Layer param is invalid");
            }
        }

        output_dims[index] = (ends[i] - begins[i] - 1) / strides[i] + 1;

        if (output_dims[index] <= 0) {
            if (status) {
                *status = Status(TNNERR_PARAM_ERR, "StrideSliceV2Layer param is invalid");
            }
        }
    }
    
    return output_dims;
}

DimsVector DimsVectorUtils::Pad(const DimsVector output_index, DimsVector input_dims, DimsVector pads,
                      int type, Status *status) {
    DimsVector input_index(output_index.size(), 0);
    if (type != 0) {
        if (status) {
            *status = Status(TNNERR_PARAM_ERR, "PadV2 type is not supported");
        }
        return input_index;
    }
    
    for (int i=0; i<input_dims.size(); i++) {
        input_index[i] = output_index[i] - pads[i];
    }
    
    return input_index;
}

bool DimsVectorUtils::IsInBox(const DimsVector index, const DimsVector shape) {
    for (int i=0; i<index.size(); i++) {
        if (index[i] < 0 || index[i] >= shape[i]) {
            return false;
        }
    }
    
    return true;
}

DimsVector DimsVectorUtils::NCHW2NHWC(DimsVector dims) {
    ASSERT(dims.size() == 4);
    const int n           = dims[0];
    const int c           = dims[1];
    const int h           = dims[2];
    const int w           = dims[3];
    std::vector<int> nhwc = {n, h, w, c};
    return nhwc;
}

DimsVector DimsVectorUtils::NHWC2NCHW(DimsVector dims) {
    ASSERT(dims.size() == 4);
    const int n           = dims[0];
    const int h           = dims[1];
    const int w           = dims[2];
    const int c           = dims[3];
    std::vector<int> nhwc = {n, c, h, w};
    return nhwc;
}

DimsVector DimsVectorUtils::IncreaseIndex(DimsVector index, const DimsVector shape, int offset) {
    if (index.size() <= 0) {
        return index;
    }
    
    int value = offset;
    for (int i=(int)index.size()-1; i>=0; i--) {
        value += index[i];
        int next_offset = 0;
        while (value >= shape[i]) {
            value -= shape[i];
            next_offset++;
        }
        index[i] = value;
        
        value = next_offset;
    }
    
    return index;
}

DimsVector DimsVectorUtils::StrideOfShape(DimsVector shape) {
    if (shape.size() <= 0) {
        return shape;
    }
    
    DimsVector stride(shape.size());
    for (int i=0; i<stride.size(); i++) {
        stride[i] = Count(shape, i+1);
    }
    return stride;
}

}  // namespace TNN_NS
