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

#include "tnn/utils/dims_function_utils.h"
#include "tnn/utils/dims_vector_utils.h"


#include <cmath>
#include <climits>

namespace TNN_NS {

DimsVector DimsFunctionUtils::Expand(DimsVector dims0, DimsVector dims1, Status *status) {
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

DimsVector DimsFunctionUtils::Upsample(const DimsVector input_dims,
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

DimsVector DimsFunctionUtils::Range(const RangeData start, const RangeData limit,
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

DimsVector DimsFunctionUtils::Reshape(const DimsVector input_dims, const DimsVector shape,
                                    const int axis, const int num_axes, Status *status) {

    int output_size = shape.size() + axis;
    DimsVector output_dims(output_size, 1);

    for(int i = 0; i < axis; ++i) {
        output_dims[i] = input_dims[i];
    }

    int infer_dim_count = 0;
    int infer_dim_pos   = -1;
    for (int i = axis, j = 0; j < num_axes; i++, j++) {
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

DimsVector DimsFunctionUtils::StrideSlice(const DimsVector input_dims,
                                        DimsVector& begins, DimsVector& ends, const DimsVector strides,
                                        const DimsVector axes, Status *status) {
    if (axes.size() != begins.size() || axes.size() != ends.size() || axes.size() != strides.size()) {
        if (status) {
            LOGE("StrideSliceV2Layer param of axes, ends, strides size is invalid\n");
            *status = Status(TNNERR_PARAM_ERR, "StrideSliceV2Layer param of axes, ends, strides size is invalid");
            return DimsVector();
        }
    }

    auto output_dims = input_dims;

    //前闭后开区间
    for (int i = 0; i < axes.size(); i++) {
        int index = axes[i];
        if (index < 0) {
            index = input_dims.size() + index;
        }
        if (input_dims.size() <= index || output_dims.size() <= index)
            continue;
        if (begins[i] < 0) {
            begins[i] += input_dims[index];
        }

        if (ends[i] == INT_MAX || ends[i] > input_dims[index]) {
            ends[i] = input_dims[index];
        } else if (ends[i] == INT_MIN) {
            ends[i] = -1;
        } else if (ends[i] < 0) {
            ends[i] += input_dims[index];
        }
        
        if (strides[i] > 0) {
            output_dims[index] = (ends[i] - begins[i] - 1) / strides[i] + 1;
        } else {
            output_dims[index] = (ends[i] - begins[i] + 1) / strides[i] + 1;
        }
        

        if (output_dims[index] <= 0) {
            if (status) {
                LOGE("StrideSliceV2Layer param is invalid\n");
                *status = Status(TNNERR_PARAM_ERR, "StrideSliceV2Layer param is invalid");
            }
        }
    }
    
    return output_dims;
}

DimsVector DimsFunctionUtils::Pad(const DimsVector output_index, DimsVector input_dims, DimsVector pads,
                      int type, Status *status) {
    DimsVector input_index(output_index.size(), 0);
    if (type != 0 && type != 1) {
        if (status) {
            *status = Status(TNNERR_PARAM_ERR, "PadV2 type is not supported");
        }
        return input_index;
    }

    if (type == 0) {
        for (int i = 0; i < input_dims.size(); i++) {
            input_index[i] = output_index[i] - pads[i];
        }
    } else if (type == 1) {
        for (int i = 0; i < input_dims.size(); i++) {
            int dst        = output_index[i];
            int pad        = pads[i];
            int input      = input_dims[i];
            input_index[i] = dst >= pad ? (dst < pad + input ? dst - pad : pad - 2 - dst + 2 * input) : pad - dst;
        }
    }
    
    return input_index;
}

bool DimsFunctionUtils::IsInBox(const DimsVector index, const DimsVector shape) {
    for (int i=0; i<index.size(); i++) {
        if (index[i] < 0 || index[i] >= shape[i]) {
            return false;
        }
    }
    
    return true;
}

DimsVector DimsFunctionUtils::IncreaseIndex(DimsVector index, const DimsVector shape, int offset) {
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

DimsVector DimsFunctionUtils::StrideOfShape(DimsVector shape) {
    if (shape.size() <= 0) {
        return shape;
    }
    
    DimsVector stride(shape.size());
    for (int i=0; i<stride.size(); i++) {
        stride[i] = DimsVectorUtils::Count(shape, i+1);
    }
    return stride;
}

DimsVector DimsFunctionUtils::Tile(const DimsVector input_dims, const DimsVector reps) {
    DimsVector output_dims = input_dims;
    if (reps.size() > input_dims.size()) {
        output_dims = reps;
    }
    for (auto index_i=(int)input_dims.size()-1, index_o=(int)output_dims.size()-1, index_r=(int)reps.size()-1;
         index_i>=0 && index_r>=0;
         index_i--,index_o--,index_r--) {
        output_dims[index_o] = input_dims[index_i] * reps[index_r];
    }
    return output_dims;
}


DimsVector DimsFunctionUtils::ModIndex(DimsVector index, const DimsVector shape) {
    for (int i=0; i<index.size() && i< shape.size(); i++) {
        index[i] %= shape[i];
    }
    return index;
}

int DimsFunctionUtils::GetDim(const DimsVector dims, const int index) {
    return dims.size() > index ? dims[index] : 1;
}

int DimsFunctionUtils::GetDimProduct(const DimsVector dims, const int start_index, const int end_index) {
    auto count = DimsVectorUtils::Count(dims, start_index, end_index);
    return count > 0? count : 1;
}

DimsVector DimsFunctionUtils::GetDimsStep(const DimsVector& dims) {
    DimsVector step_dims;
    for(int i = 0; i < dims.size(); ++i) {
        step_dims.push_back(DimsVectorUtils::Count(dims, i+1));
    }
    return step_dims;
}

}  // namespace TNN_NS
