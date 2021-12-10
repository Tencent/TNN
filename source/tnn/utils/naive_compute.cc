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

#include "tnn/utils/naive_compute.h"

#include <cstring>
#include <type_traits>
#include <queue>

#include "math.h"
#include "tnn/core/macro.h"
#include "tnn/interpreter/layer_param.h"
#include "tnn/utils/bbox_util.h"
#include "tnn/utils/bfp16.h"
#include "tnn/utils/dims_utils.h"
#include "tnn/utils/omp_utils.h"
#include "tnn/utils/half_utils_inner.h"

namespace TNN_NS {

int8_t float2int8(float val) {
    return static_cast<int8_t>(MAX(MIN(val + (val >= 0.f ? 0.5f : -0.5f), 127.0f), -128.0f));
}

uint8_t float2uint8(float val) {
    return static_cast<uint8_t>(MAX(MIN(val + (val >= 0.f ? 0.5f : -0.5f), 255.0f), 0.0f));
}

int8_t half2int8(fp16_t val) {
    return static_cast<int8_t>(MAX(MIN(val + (val >= 0.f ? 0.5f : -0.5f), 127.0f), -128.0f));
}

uint8_t half2uint8(fp16_t val) {
    return static_cast<uint8_t>(MAX(MIN(val + (val >= 0.f ? 0.5f : -0.5f), 255.0f), 0.0f));
}

static inline int start_index(int a, int b, int c) {
    return (int)std::floor((float)(a * c) / b);
}

static inline int end_index(int a, int b, int c) {
    return (int)std::ceil((float)((a + 1) * c) / b);
}

template <typename T, typename Tacc>
void NaiveAdaptivePooling(T *input_data, T *output_data, DimsVector dims_input, DimsVector dims_output, int pool_type) {
    bool is_1d             = dims_input.size() == 3;
    const int channels     = is_1d ? dims_input[0] : dims_input[0] * dims_input[1];
    const int input_height = is_1d ? dims_input[1] : dims_input[2];
    const int input_width  = is_1d ? dims_input[2] : dims_input[3];
    int64_t output_height  = is_1d ? dims_output[1] : dims_output[2];
    int64_t output_width   = is_1d ? dims_output[2] : dims_output[3];

    for (int c = 0; c < channels; c++) {
        T *input_ptr  = input_data + c * input_height * input_width;
        T *output_ptr = output_data + c * output_height * output_width;

        for (int oh = 0; oh < output_height; oh++) {
            int ih0 = start_index(oh, output_height, input_height);
            int ih1 = end_index(oh, output_height, input_height);
            int kh  = ih1 - ih0;

            for (int ow = 0; ow < output_width; ow++) {
                int iw0 = start_index(ow, output_width, input_width);
                int iw1 = end_index(ow, output_width, input_width);
                int kw  = iw1 - iw0;

                // compute local average
                if (pool_type == 1) {
                    T sum = 0;
                    for (int ih = ih0; ih < ih1; ih++) {
                        for (int iw = iw0; iw < iw1; iw++) {
                            sum += input_ptr[ih * input_width + iw];
                        }
                    }
                    output_ptr[oh * output_width + ow] = sum / kh / kw;
                }
            }
        }
    }
}

// initialize the NaiveAdaptivePooling FUNTION with float
template void NaiveAdaptivePooling<float, float>(float *input_data, float *output_data, DimsVector dims_input,
                                                 DimsVector dims_output, int pool_type);

/*
 * Computes max pooling or average pooling
 * blob data format must be NCHW
 */
template <typename T, typename Tacc>
void NaivePooling(T *input_ptr, T *output_ptr, DimsVector dims_input, DimsVector dims_output, int stride_y,
                  int stride_x, int kernel_y, int kernel_x, int pad_y, int pad_x, int pool_type) {
    auto input_width = dims_input[3], input_height = dims_input[2];
    auto output_width = dims_output[3], output_height = dims_output[2], output_channel = dims_output[1];
    for (int n = 0; n < dims_output[0]; n++) {
        T *in_current_batch = input_ptr + n * input_width * input_height * output_channel;
        T *ou_current_batch = output_ptr + n * output_width * output_height * output_channel;
        for (int c = 0; c < output_channel; c++) {
            for (int h = 0; h < output_height; h++) {
                for (int w = 0; w < output_width; w++) {
                    // value is accumulated in the type Tacc
                    // which is float for both float and bfp16
                    Tacc calc_val;
                    if (std::is_same<T, float>::value || std::is_same<T, bfp16_t>::value) {
                        calc_val = static_cast<Tacc>(-FLT_MAX);
                    } else if (std::is_same<T, int8_t>::value) {
                        calc_val = static_cast<Tacc>(-INT8_MAX);
                    }
                    calc_val = pool_type == 0 ? calc_val : 0;

                    T cur_val = static_cast<T>(0);

                    int hstart       = h * stride_y - pad_y;
                    int wstart       = w * stride_x - pad_x;
                    int hend         = std::min(hstart + kernel_y, input_height);
                    int wend         = std::min(wstart + kernel_x, input_width);
                    hstart           = std::max(hstart, 0);
                    wstart           = std::max(wstart, 0);
                    int kernel_count = (hend - hstart) * (wend - wstart);

                    for (int inh = hstart; inh < hend; ++inh) {
                        for (int inw = wstart; inw < wend; ++inw) {
                            cur_val = in_current_batch[c * input_height * input_width + inh * input_width + inw];

                            if (pool_type == 0) {  // max pooling
                                calc_val = std::max((Tacc)cur_val, calc_val);
                            } else {
                                // pool_type ==1 for average pooling
                                calc_val += cur_val;
                            }
                        }
                    }

                    if (pool_type == 0) {  // max pooling
                        calc_val = std::max((Tacc)cur_val, calc_val);
                    } else {
                        // average pooling
                        calc_val = calc_val / kernel_count;
                    }

                    ou_current_batch[c * output_height * output_width + h * output_width + w] =
                        static_cast<T>(calc_val);
                }
            }
        }
    }
}

// initialize the NaivePooling FUNTION with float
template void NaivePooling<float, float>(float *input_ptr, float *output_ptr, DimsVector dims_input,
                                         DimsVector dims_output, int stride_y, int stride_x, int kernel_y, int kernel_x,
                                         int pad_y, int pad_x, int pool_type);

// initialize the NaivePooling FUNTION with bfp16
template void NaivePooling<bfp16_t, float>(bfp16_t *input_ptr, bfp16_t *output_ptr, DimsVector dims_input,
                                           DimsVector dims_output, int stride_y, int stride_x, int kernel_y,
                                           int kernel_x, int pad_y, int pad_x, int pool_type);

// initialize the NaivePooling FUNTION with int8
template void NaivePooling<int8_t, int32_t>(int8_t *input_ptr, int8_t *output_ptr, DimsVector dims_input,
                                            DimsVector dims_output, int stride_y, int stride_x, int kernel_y,
                                            int kernel_x, int pad_y, int pad_x, int pool_type);

/*
 * Computes max pooling or average 3d pooling
 * blob data format must be NCDHW
 */
template <typename T, typename Tacc>
void NaivePooling3D(T *input_ptr, T *output_ptr, DimsVector dims_input, DimsVector dims_output,
                    int stride_d, int stride_y, int stride_x, int kernel_d, int kernel_y, int kernel_x,
                    int pad_d, int pad_y, int pad_x, int pool_type) {
    auto input_width = dims_input[4], input_height = dims_input[3], input_depth = dims_input[2];
    auto output_width = dims_output[4], output_height = dims_output[3];
    auto output_depth = dims_output[2], output_channel = dims_output[1];
    for (int n = 0; n < dims_output[0]; n++) {
        T *in_current_batch = input_ptr + n * input_width * input_height * input_depth * output_channel;
        T *ou_current_batch = output_ptr + n * output_width * output_height * output_depth * output_channel;
        for (int c = 0; c < output_channel; c++) {
            for (int d = 0; d < output_depth; d++) {
                for (int h = 0; h < output_height; h++) {
                    for (int w = 0; w < output_width; w++) {
                        // value is accumulated in the type Tacc
                        // which is float for both float and bfp16
                        Tacc calc_val;
                        if (std::is_same<T, float>::value || std::is_same<T, bfp16_t>::value) {
                            calc_val = static_cast<Tacc>(-FLT_MAX);
                        } else if (std::is_same<T, int8_t>::value) {
                            calc_val = static_cast<Tacc>(-INT8_MAX);
                        }
                        calc_val = pool_type == 0 ? calc_val : 0;

                        T cur_val = static_cast<T>(0);

                        int dstart       = d * stride_d - pad_d;
                        int hstart       = h * stride_y - pad_y;
                        int wstart       = w * stride_x - pad_x;
                        int dend         = std::min(dstart + kernel_d, input_depth);
                        int hend         = std::min(hstart + kernel_y, input_height);
                        int wend         = std::min(wstart + kernel_x, input_width);
                        dstart           = std::max(dstart, 0);
                        hstart           = std::max(hstart, 0);
                        wstart           = std::max(wstart, 0);
                        int kernel_count = (dend - dstart) * (hend - hstart) * (wend - wstart);

                        for (int ind = dstart; ind < dend; ++ind) {
                            for (int inh = hstart; inh < hend; ++inh) {
                                for (int inw = wstart; inw < wend; ++inw) {
                                    cur_val =
                                        in_current_batch[c * input_height * input_width * input_depth + 
                                                            ind * input_height * input_width + inh * input_width + inw];

                                    if (pool_type == 0) {  // max pooling
                                        calc_val = std::max((Tacc)cur_val, calc_val);
                                    } else {
                                        // pool_type ==1 for average pooling
                                        calc_val += cur_val;
                                    }
                                }
                            }
                        }

                        if (pool_type == 0) {  // max pooling
                            calc_val = std::max((Tacc)cur_val, calc_val);
                        } else {
                            // average pooling
                            calc_val = calc_val / kernel_count;
                        }

                        ou_current_batch[c * output_height * output_width * output_depth
                                            + d * output_height * output_width
                                            + h * output_width + w] =
                            static_cast<T>(calc_val);
                    }
                }
            }
        }
    }
}

// initialize the NaivePooling FUNTION with float
template void NaivePooling3D<float, float>(float *input_ptr, float *output_ptr, DimsVector dims_input, DimsVector dims_output,
                    int stride_d, int stride_y, int stride_x, int kernel_d, int kernel_y, int kernel_x,
                    int pad_d, int pad_y, int pad_x, int pool_type);

// initialize the NaivePooling FUNTION with bfp16
template void NaivePooling3D<bfp16_t, float>(bfp16_t *input_ptr, bfp16_t *output_ptr, DimsVector dims_input, DimsVector dims_output,
                    int stride_d, int stride_y, int stride_x, int kernel_d, int kernel_y, int kernel_x,
                    int pad_d, int pad_y, int pad_x, int pool_type);

// initialize the NaivePooling FUNTION with int8
template void NaivePooling3D<int8_t, int32_t>(int8_t *input_ptr, int8_t *output_ptr, DimsVector dims_input, DimsVector dims_output,
                    int stride_d, int stride_y, int stride_x, int kernel_d, int kernel_y, int kernel_x,
                    int pad_d, int pad_y, int pad_x, int pool_type);

/*
 * Full Connected funtion
 * blob data format is required to be NCHW
 */
template <typename T>
void NaiveFC(T *input_ptr, T *output_ptr, T *weight_data, float *bias, DimsVector dims_input, DimsVector dims_output) {
    int ip_dim_in = DimsVectorUtils::Count(dims_input, 1);
    for (int n = 0; n < dims_output[0]; ++n) {
        T *in_current_batch = input_ptr + n * ip_dim_in;
        T *ou_current_batch = output_ptr + n * dims_output[1];
        OMP_PARALLEL_FOR_
        for (int oc = 0; oc < dims_output[1]; ++oc) {
            float acc = 0;
            for (int ic = 0; ic < ip_dim_in; ++ic) {
                acc += float(static_cast<T *>(weight_data)[oc * ip_dim_in + ic]) * float(in_current_batch[ic]);
            }
            if (bias)
                acc += bias[oc];
            ou_current_batch[oc] = acc;
        }
    }
}

template void NaiveFC(float *input_ptr, float *output_ptr, float *weight_data, float *bias, DimsVector dims_input,
                      DimsVector dims_output);

template void NaiveFC(bfp16_t *input_ptr, bfp16_t *output_ptr, bfp16_t *weight_data, float *bias, DimsVector dims_input,
                      DimsVector dims_output);

// specialize for the case data_type=int8
void NaiveFC(void *input_ptr, void *output_ptr, void *weight_data, float *scale, int scale_len, void *bias,
            DimsVector dims_input, DimsVector dims_output) {
    int ip_dim_in = DimsVectorUtils::Count(dims_input, 1);
    for (int n = 0; n < dims_output[0]; ++n) {
        int8_t *in_current_batch = static_cast<int8_t *>(input_ptr) + n * ip_dim_in;
        int8_t *ou_current_batch = static_cast<int8_t *>(output_ptr) + n * dims_output[1];
        OMP_PARALLEL_FOR_
        for (int oc = 0; oc < dims_output[1]; ++oc) {
            float cur_scale = scale_len == 1 ? scale[0] : scale[oc];
            int32_t acc     = 0;
            for (int ic = 0; ic < ip_dim_in; ++ic) {
                acc += static_cast<int8_t *>(weight_data)[oc * ip_dim_in + ic] * in_current_batch[ic];
            }
            if (bias)
                acc += static_cast<int32_t *>(bias)[oc];
            ou_current_batch[oc] = float2int8(acc * cur_scale);
        }
    }
}
void NaiveFCBias(void *input_ptr, void *output_ptr, void *weight_data, float *scale, int scale_len, void *bias,
                 void *zero_point_w_ptr, void *zero_point_i_ptr, void *zero_point_o_ptr, int zero_point_len_w,
                 int zero_point_len_i, int zero_point_len_o, DimsVector dims_input, DimsVector dims_output) {
    int8_t *zero_point_handle_i = static_cast<int8_t *>(zero_point_i_ptr);
    int8_t *zero_point_handle_w = static_cast<int8_t *>(zero_point_w_ptr);
    int8_t *zero_point_handle_o = static_cast<int8_t *>(zero_point_o_ptr);
    int ip_dim_in               = DimsVectorUtils::Count(dims_input, 1);
    int ip_dim_hw               = DimsVectorUtils::Count(dims_input, 2);
    for (int n = 0; n < dims_output[0]; ++n) {
        int8_t *in_current_batch = static_cast<int8_t *>(input_ptr) + n * ip_dim_in;
        int8_t *ou_current_batch = static_cast<int8_t *>(output_ptr) + n * dims_output[1];
        OMP_PARALLEL_FOR_
        for (int oc = 0; oc < dims_output[1]; ++oc) {
            float cur_scale         = scale_len == 1 ? scale[0] : scale[oc];
            float cur_bias_output   = zero_point_len_o == 1 ? zero_point_handle_o[0] : zero_point_handle_o[oc];
            int8_t cur_zero_point_w = zero_point_len_w == 1 ? zero_point_handle_w[0] : zero_point_handle_w[oc];
            int32_t acc             = 0;
            for (int ic = 0; ic < ip_dim_in; ++ic) {
                int ichannel = ic / ip_dim_hw;
                int8_t cur_zero_point_i = zero_point_len_i == 1 ? zero_point_handle_i[0] : zero_point_handle_i[ichannel];
                acc +=
                    static_cast<int8_t *>(weight_data)[oc * ip_dim_in + ic] * in_current_batch[ic] -
                    static_cast<int32_t>(in_current_batch[ic] * cur_zero_point_w) -
                    static_cast<int32_t>(cur_zero_point_i * static_cast<int8_t *>(weight_data)[oc * ip_dim_in + ic]) +
                    static_cast<int32_t>(cur_zero_point_i * cur_zero_point_w);
            }
            if (bias)
                acc += static_cast<int32_t *>(bias)[oc];
            ou_current_batch[oc] = float2int8(acc * cur_scale + cur_bias_output);
        }
    }
}

template <typename Tacc>
void FloatActivate(Tacc &result, const int activation_type) {
    if (activation_type == ActivationType_ReLU) {
        result = static_cast<Tacc>(result > 0.0f ? result : 0.0f);
    } else if (activation_type == ActivationType_ReLU6) {
        if (result > 6.0f) {
            result = static_cast<Tacc>(6.0f);
        } else if (result < 0.0f) {
            result = static_cast<Tacc>(0.0f);
        }
    } else if(activation_type == ActivationType_SIGMOID_MUL) {
        result = 1.0f / (1.0f + exp(-result)) * result;
    }
}

template <typename Tin, typename Tw, typename Tacc, typename Tout>
void NaiveConv1D(void *input_ptr, void *output_ptr, void *weight_ptr, void *bias, DimsVector dims_input,
                 DimsVector dims_output, int stride, int kernel_size, int pad, int group, int dilation,
                 int activation_type, float *scale, int scale_len, int fusion_type, void *add_input, float *add_scale) {
    Tin *input_data               = static_cast<Tin *>(input_ptr);
    Tw *weight_data               = static_cast<Tw *>(weight_ptr);
    Tout *output_data             = static_cast<Tout *>(output_ptr);
    Tacc *bias_data               = static_cast<Tacc *>(bias);
    int number                    = dims_output[0];
    int output_channel            = dims_output[1];
    int output_height             = dims_output[2];
    int input_channel             = dims_input[1];
    int input_height              = dims_input[2];
    int output_channels_per_group = output_channel / group;
    int input_channels_per_group  = input_channel / group;

    OMP_PARALLEL_FOR_
    for (int n = 0; n < number; ++n) {
        for (int g = 0; g < group; ++g) {
            int output_c_start = g * output_channels_per_group;
            int output_c_end   = (g + 1) * output_channels_per_group;
            int input_c_start  = g * input_channels_per_group;
            int input_c_end    = (g + 1) * input_channels_per_group;
            int weights_start  = g * output_channels_per_group * input_channels_per_group * kernel_size;
            for (int output_c = output_c_start; output_c < output_c_end; ++output_c) {
                for (int h = 0; h < output_height; ++h) {
                    int input_h_start = h * stride - pad;
                    Tacc result       = static_cast<Tacc>(0.0f);
                    for (int kernel_h = 0; kernel_h < kernel_size; ++kernel_h) {
                        int input_h = input_h_start + kernel_h * dilation;
                        if (input_h < 0 || input_h >= input_height) {
                            continue;
                        }
                        for (int input_c = input_c_start; input_c < input_c_end; ++input_c) {
                            int input_position = (n * input_channel + input_c) * input_height + input_h;
                            int weight_position =
                                weights_start +
                                ((output_c - output_c_start) * input_channels_per_group + input_c - input_c_start) *
                                kernel_size +
                                kernel_h;
                            auto ip = input_data[input_position];
                            auto wd = weight_data[weight_position];
                            result += input_data[input_position] * weight_data[weight_position];
                        }
                    }

                    int output_position = (n * output_channel + output_c) * output_height + h;
                    if (bias_data) {
                        result += bias_data[output_c];
                    }
                    if (sizeof(Tin) > 1) {  // float
                        FloatActivate(result, activation_type);
                        output_data[output_position] = result;
                    } else {
                        int scaleidx = scale_len == 1 ? 0 : output_c;
                        float val    = result * scale[scaleidx];
                        if (fusion_type == FusionType_Conv_Add_Activation) {
                            val += static_cast<Tin *>(add_input)[output_position] * add_scale[output_c];
                        }
                        if (activation_type == ActivationType_ReLU) {
                            val = std::max(0.0f, val);
                        }
                        if (fusion_type == FusionType_Conv_Activation_Add) {
                            val += static_cast<Tin *>(add_input)[output_position] * add_scale[output_c];
                        }
                        output_data[output_position] = float2int8(val);
                    }
                }
            }
        }
    }
}

template void NaiveConv1D<float, float, float, float>(void *input_ptr, void *output_ptr, void *weight_ptr, void *bias,
                                                      DimsVector dims_input, DimsVector dims_output, int stride,
                                                      int kernel_size, int pad, int group, int dilation,
                                                      int activation_type, float *scale, int scale_len, int fusion_type,
                                                      void *add_input, float *add_scale);

/*
 * convolution funtion
 * input & output data_format is NCHW
 * depthwise is supported
 */
template <typename Tin, typename Tw, typename Tacc, typename Tout>
void NaiveConv(void *input_ptr, void *output_ptr, void *weight_ptr, void *bias, DimsVector dims_input,
               DimsVector dims_output, int stride_y, int stride_x, int kernel_size_y, int kernel_size_x, int pad_y,
               int pad_x, int group, int dilation, int activation_type, float *weight_scale, int weight_scale_len,
               int8_t *relu6_max, int relu6_max_len, int fusion_type, void *add_input, float *add_scale) {
    Tin *input_data               = static_cast<Tin *>(input_ptr);
    Tw *weight_data               = static_cast<Tw *>(weight_ptr);
    Tout *output_data             = static_cast<Tout *>(output_ptr);
    Tacc *bias_data               = static_cast<Tacc *>(bias);
    int number                    = dims_output[0];
    int output_channel            = dims_output[1];
    int output_height             = dims_output[2];
    int output_width              = dims_output[3];
    int input_channel             = dims_input[1];
    int input_height              = dims_input[2];
    int input_width               = dims_input[3];
    int output_channels_per_group = output_channel / group;
    int input_channels_per_group  = input_channel / group;

    OMP_PARALLEL_FOR_
    for (int n = 0; n < number; ++n) {
        for (int g = 0; g < group; ++g) {
            int output_c_start = g * output_channels_per_group;
            int output_c_end   = (g + 1) * output_channels_per_group;
            int input_c_start  = g * input_channels_per_group;
            int input_c_end    = (g + 1) * input_channels_per_group;
            int weights_start =
                g * output_channels_per_group * input_channels_per_group * kernel_size_x * kernel_size_y;
            for (int output_c = output_c_start; output_c < output_c_end; ++output_c) {
                for (int h = 0; h < output_height; ++h) {
                    int input_h_start = h * stride_y - pad_y;
                    for (int w = 0; w < output_width; ++w) {
                        int input_w_start = w * stride_x - pad_x;
                        Tacc result       = static_cast<Tacc>(0.0f);
                        for (int kernel_h = 0; kernel_h < kernel_size_y; ++kernel_h) {
                            int input_h = input_h_start + kernel_h * dilation;
                            if (input_h < 0 || input_h >= input_height) {
                                continue;
                            }
                            for (int kernel_w = 0; kernel_w < kernel_size_x; ++kernel_w) {
                                int input_w = input_w_start + kernel_w * dilation;
                                if (input_w < 0 || input_w >= input_width) {
                                    continue;
                                }
                                for (int input_c = input_c_start; input_c < input_c_end; ++input_c) {
                                    int input_position =
                                        ((n * input_channel + input_c) * input_height + input_h) * input_width +
                                        input_w;
                                    int weight_position = weights_start +
                                                          (((output_c - output_c_start) * input_channels_per_group +
                                                            input_c - input_c_start) *
                                                               kernel_size_y +
                                                           kernel_h) *
                                                              kernel_size_x +
                                                          kernel_w;
                                    result += input_data[input_position] * weight_data[weight_position];
                                }
                            }
                        }

                        int output_position = ((n * output_channel + output_c) * output_height + h) * output_width + w;
                        if (bias_data) {
                            result += bias_data[output_c];
                        }
                        if (sizeof(Tin) > 1) {  // float
                            FloatActivate(result, activation_type);
                            output_data[output_position] = result;
                        } else {
                            int scale_idx = weight_scale_len == 1 ? 0 : output_c;
                            float val    = result * weight_scale[scale_idx];
                            if (fusion_type == FusionType_Conv_Add_Activation) {
                                val += static_cast<Tin *>(add_input)[output_position] * add_scale[output_c];
                            }
                            if (activation_type == ActivationType_ReLU) {
                                val = std::max(0.0f, val);
                            } else if (activation_type == ActivationType_ReLU6) {
                                int relu6_max_idx = relu6_max_len == 1 ? 0:output_c;
                                int8_t res = std::min(float2int8(val), relu6_max[relu6_max_idx]);
                                res = std::max((int8_t)0, res);
                                output_data[output_position] = res;
                                continue;
                            }
                            if (fusion_type == FusionType_Conv_Activation_Add) {
                                val += static_cast<Tin *>(add_input)[output_position] * add_scale[output_c];
                            }
                            output_data[output_position] = float2int8(val);
                        }
                    }
                }
            }
        }
    }
}
template <typename Tin, typename Tw, typename Tacc, typename Tout>
void NaiveConvBias(void *input_ptr, void *output_ptr, void *weight_ptr, void *bias, DimsVector dims_input,
                   DimsVector dims_output, int stride_y, int stride_x, int kernel_size_y, int kernel_size_x, int pad_y,
                   int pad_x, int group, int dilation, int activation_type, float *weight_scale, int weight_scale_len,
                   void *zero_point_w_ptr, int zero_point_len_w, void *zero_point_i_ptr, int zero_point_len_i,
                   void *zero_point_o_ptr, int zero_point_len_o, void *weight_x_bias_ptr, int8_t *relu6_max,
                   int relu6_max_len, int fusion_type, void *add_input, float *add_scale, void *add_bias_input) {
    Tin *input_data               = static_cast<Tin *>(input_ptr);
    Tw *weight_data               = static_cast<Tw *>(weight_ptr);
    Tout *output_data             = static_cast<Tout *>(output_ptr);
    Tacc *bias_data               = static_cast<Tacc *>(bias);
    int number                    = dims_output[0];
    int output_channel            = dims_output[1];
    int output_height             = dims_output[2];
    int output_width              = dims_output[3];
    int input_channel             = dims_input[1];
    int input_height              = dims_input[2];
    int input_width               = dims_input[3];
    int output_channels_per_group = output_channel / group;
    int input_channels_per_group  = input_channel / group;
    Tin *zero_point_handle_i      = static_cast<Tin *>(zero_point_i_ptr);
    Tw *zero_point_handle_w       = static_cast<Tw *>(zero_point_w_ptr);
    Tout *zero_point_handle_o     = static_cast<Tout *>(zero_point_o_ptr);
    Tacc *buffer_weight_x_bias    = static_cast<Tacc *>(weight_x_bias_ptr);
    Tin *add_bias_i               = static_cast<Tin *>(add_bias_input);

    OMP_PARALLEL_FOR_
    for (int n = 0; n < number; ++n) {
        for (int g = 0; g < group; ++g) {
            int output_c_start = g * output_channels_per_group;
            int output_c_end   = (g + 1) * output_channels_per_group;
            int input_c_start  = g * input_channels_per_group;
            int input_c_end    = (g + 1) * input_channels_per_group;
            int weights_start =
                g * output_channels_per_group * input_channels_per_group * kernel_size_x * kernel_size_y;
            for (int output_c = output_c_start; output_c < output_c_end; ++output_c) {
                int scale_idx = weight_scale_len == 1 ? 0 : output_c;
                int weight_bias_idx = zero_point_len_w == 1 ? 0 : output_c;
                int output_bias_idx = zero_point_len_o == 1 ? 0 : output_c;
                for (int h = 0; h < output_height; ++h) {
                    int input_h_start = h * stride_y - pad_y;
                    for (int w = 0; w < output_width; ++w) {
                        int input_w_start = w * stride_x - pad_x;
                        Tacc result       = static_cast<Tacc>(0.0f);
                        int output_position = ((n * output_channel + output_c) * output_height + h) * output_width + w;
                        for (int kernel_h = 0; kernel_h < kernel_size_y; ++kernel_h) {
                            int input_h = input_h_start + kernel_h * dilation;
                            bool pad_flag_h = false;
                            if (input_h < 0 || input_h >= input_height) {
                                pad_flag_h = true;
                            }
                            for (int kernel_w = 0; kernel_w < kernel_size_x; ++kernel_w) {
                                int input_w = input_w_start + kernel_w * dilation;
                                bool pad_flag_w = false;
                                if (input_w < 0 || input_w >= input_width) {
                                    pad_flag_w = true;
                                }
                                for (int input_c = input_c_start; input_c < input_c_end; ++input_c) {
                                    int weight_position = weights_start +
                                                          (((output_c - output_c_start) * input_channels_per_group +
                                                            input_c - input_c_start) *
                                                               kernel_size_y +
                                                           kernel_h) *
                                                              kernel_size_x +
                                                          kernel_w;
                                    if (pad_flag_h || pad_flag_w) {
                                        int input_bias_idx = zero_point_len_i == 1 ? 0 : input_c;
                                        result += static_cast<Tacc>(zero_point_handle_i[input_bias_idx] *
                                                                    weight_data[weight_position]) -
                                                  static_cast<Tacc>(zero_point_handle_i[input_bias_idx] *
                                                                    zero_point_handle_w[weight_bias_idx]);
                                    } else {
                                        int input_position =
                                            ((n * input_channel + input_c) * input_height + input_h) * input_width +
                                            input_w;
                                        result += input_data[input_position] * weight_data[weight_position] -
                                                  static_cast<Tacc>(input_data[input_position] *
                                                                    zero_point_handle_w[weight_bias_idx]);
                                    }
                                }
                            }
                        }
                        result += buffer_weight_x_bias[output_c];
                        if (bias_data) {
                            result += bias_data[output_c];
                        }
                        if (sizeof(Tin) > 1) {  // float
                            FloatActivate(result, activation_type);
                            output_data[output_position] = result;
                        } else {
                            float val = result * weight_scale[scale_idx];
                            if (fusion_type == FusionType_Conv_Add_Activation) {
                                val += static_cast<Tin *>(add_input)[output_position] * add_scale[output_c] -
                                       add_bias_i[output_bias_idx] * add_scale[output_c];
                            }
                            if (activation_type == ActivationType_ReLU) {
                                val = std::max(0.0f, val);
                            } else if (activation_type == ActivationType_ReLU6) {
                                int relu6_max_idx            = relu6_max_len == 1 ? 0 : output_c;
                                int8_t res                   = std::min(float2int8(val), relu6_max[relu6_max_idx]);
                                res                          = std::max((int8_t)0, res);
                                output_data[output_position] = res;
                                continue;
                            }
                            if (fusion_type == FusionType_Conv_Activation_Add) {
                                val += static_cast<Tin *>(add_input)[output_position] * add_scale[output_c] -
                                       add_bias_i[output_bias_idx] * add_scale[output_c];
                            }
                            val += static_cast<Tacc>(zero_point_handle_o[output_bias_idx]);
                            output_data[output_position] = float2int8(val);
                        }
                    }
                }
            }
        }
    }
}
template <typename Tin, typename Tw, typename Tacc, typename Tout>
void NaiveConvBias(void *input_ptr, void *output_ptr, void *weight_ptr, void *bias, DimsVector dims_input,
                   DimsVector dims_output, int stride_y, int stride_x, int kernel_size_y, int kernel_size_x, int pad_y,
                   int pad_x, int group, int dilation, int activation_type, float *weight_scale, int weight_scale_len,
                   void *zero_point_w_ptr, int zero_point_len_w, void *zero_point_i_ptr, int zero_point_len_i,
                   void *zero_point_o_ptr, int zero_point_len_o, int8_t *relu6_max, int relu6_max_len, int fusion_type,
                   void *add_input, float *add_scale, void *add_bias_input) {
    Tin *input_data               = static_cast<Tin *>(input_ptr);
    Tw *weight_data               = static_cast<Tw *>(weight_ptr);
    Tout *output_data             = static_cast<Tout *>(output_ptr);
    Tacc *bias_data               = static_cast<Tacc *>(bias);
    int number                    = dims_output[0];
    int output_channel            = dims_output[1];
    int output_height             = dims_output[2];
    int output_width              = dims_output[3];
    int input_channel             = dims_input[1];
    int input_height              = dims_input[2];
    int input_width               = dims_input[3];
    int output_channels_per_group = output_channel / group;
    int input_channels_per_group  = input_channel / group;
    Tin *zero_point_handle_i      = static_cast<Tin *>(zero_point_i_ptr);
    Tw *zero_point_handle_w       = static_cast<Tw *>(zero_point_w_ptr);
    Tout *zero_point_handle_o     = static_cast<Tout *>(zero_point_o_ptr);
    Tin *add_bias_i               = static_cast<Tin *>(add_bias_input);

    OMP_PARALLEL_FOR_
    for (int n = 0; n < number; ++n) {
        for (int g = 0; g < group; ++g) {
            int output_c_start = g * output_channels_per_group;
            int output_c_end   = (g + 1) * output_channels_per_group;
            int input_c_start  = g * input_channels_per_group;
            int input_c_end    = (g + 1) * input_channels_per_group;
            int weights_start =
                g * output_channels_per_group * input_channels_per_group * kernel_size_x * kernel_size_y;
            for (int output_c = output_c_start; output_c < output_c_end; ++output_c) {
                int scale_idx = weight_scale_len == 1 ? 0 : output_c;
                int weight_bias_idx = zero_point_len_w == 1 ? 0 : output_c;
                int output_bias_idx = zero_point_len_o == 1 ? 0 : output_c;
                for (int h = 0; h < output_height; ++h) {
                    int input_h_start = h * stride_y - pad_y;
                    for (int w = 0; w < output_width; ++w) {
                        int input_w_start   = w * stride_x - pad_x;
                        Tacc result         = static_cast<Tacc>(0.0f);
                        int output_position = ((n * output_channel + output_c) * output_height + h) * output_width + w;
                        for (int kernel_h = 0; kernel_h < kernel_size_y; ++kernel_h) {
                            int input_h = input_h_start + kernel_h * dilation;
                            if (input_h < 0 || input_h >= input_height) {
                                continue;
                            }
                            for (int kernel_w = 0; kernel_w < kernel_size_x; ++kernel_w) {
                                int input_w = input_w_start + kernel_w * dilation;
                                if (input_w < 0 || input_w >= input_width) {
                                    continue;
                                }
                                for (int input_c = input_c_start; input_c < input_c_end; ++input_c) {
                                    int input_position =
                                        ((n * input_channel + input_c) * input_height + input_h) * input_width +
                                        input_w;
                                    int weight_position = weights_start +
                                                          (((output_c - output_c_start) * input_channels_per_group +
                                                            input_c - input_c_start) *
                                                               kernel_size_y +
                                                           kernel_h) *
                                                              kernel_size_x +
                                                          kernel_w;
                                    int input_bias_idx = zero_point_len_i == 1 ? 0 : input_c;
                                    result +=
                                        input_data[input_position] * weight_data[weight_position] -
                                        static_cast<Tacc>(input_data[input_position] * zero_point_handle_w[weight_bias_idx]) -
                                        static_cast<Tacc>(zero_point_handle_i[input_bias_idx] * weight_data[weight_position]) +
                                        static_cast<Tacc>(zero_point_handle_i[input_bias_idx] *
                                                          zero_point_handle_w[weight_bias_idx]);
                                }
                            }
                        }

                        if (bias_data) {
                            result += bias_data[output_c];
                        }
                        if (sizeof(Tin) > 1) {  // float
                            FloatActivate(result, activation_type);
                            output_data[output_position] = result;
                        } else {
                            float val = result * weight_scale[scale_idx];
                            if (fusion_type == FusionType_Conv_Add_Activation) {
                                val += static_cast<Tin *>(add_input)[output_position] * add_scale[output_c] -
                                       add_bias_i[output_bias_idx] * add_scale[output_c];
                            }
                            if (activation_type == ActivationType_ReLU) {
                                val = std::max(0.0f, val);
                            } else if (activation_type == ActivationType_ReLU6) {
                                int relu6_max_idx = relu6_max_len == 1 ? 0:output_c;
                                int8_t res = std::min(float2int8(val), relu6_max[relu6_max_idx]);
                                res = std::max((int8_t)0, res);
                                output_data[output_position] = res;
                                continue;
                            }
                            if (fusion_type == FusionType_Conv_Activation_Add) {
                                val += static_cast<Tin *>(add_input)[output_position] * add_scale[output_c] -
                                       add_bias_i[output_bias_idx] * add_scale[output_c];
                            }
                            val += static_cast<Tin>(zero_point_handle_o[output_bias_idx]);
                            output_data[output_position] = float2int8(val);
                        }
                    }
                }
            }
        }
    }
}

template void NaiveConvBias<int8_t, int8_t, int32_t, int8_t>(
    void *input_ptr, void *output_ptr, void *weight_ptr, void *bias, DimsVector dims_input, DimsVector dims_output,
    int stride_y, int stride_x, int kernel_size_y, int kernel_size_x, int pad_y, int pad_x, int group, int dilation,
    int activation_type, float *weight_scale, int weight_scale_len, void *zero_point_w_ptr, int zero_point_len_w,
    void *zero_point_i_ptr, int zero_point_len_i, void *zero_point_o_ptr, int zero_point_len_o, int8_t *relu6_max,
    int relu6_max_len, int fusion_type, void *add_input, float *add_scale, void *add_bias_input);
template void NaiveConvBias<int8_t, int8_t, int32_t, int8_t>(
    void *input_ptr, void *output_ptr, void *weight_ptr, void *bias, DimsVector dims_input, DimsVector dims_output,
    int stride_y, int stride_x, int kernel_size_y, int kernel_size_x, int pad_y, int pad_x, int group, int dilation,
    int activation_type, float *weight_scale, int weight_scale_len, void *zero_point_w_ptr, int zero_point_len_w,
    void *zero_point_i_ptr, int zero_point_len_i, void *zero_point_o_ptr, int zero_point_len_o, void *weight_x_bias_ptr,
    int8_t *relu6_max, int relu6_max_len, int fusion_type, void *add_input, float *add_scale, void *add_bias_input);

template void NaiveConv<float, float, float, float>(void *input_ptr, void *output_ptr, void *weight_ptr, void *bias,
                                                    DimsVector dims_input, DimsVector dims_output, int stride_y,
                                                    int stride_x, int kernel_size_y, int kernel_size_x, int pad_y,
                                                    int pad_x, int group, int dilation, int activation_type,
                                                    float *weight_scale, int weight_scale_len, int8_t *relu6_max,
                                                    int relu6_max_len, int fusion_type, void *add_input,
                                                    float *add_scale);

template void NaiveConv<int8_t, int8_t, int32_t, int8_t>(
    void *input_ptr, void *output_ptr, void *weight_ptr, void *bias, DimsVector dims_input, DimsVector dims_output,
    int stride_y, int stride_x, int kernel_size_y, int kernel_size_x, int pad_y, int pad_x, int group, int dilation,
    int activation_type, float *weight_scale, int weight_scale_len,  int8_t *relu6_max, int relu6_max_len,
    int fusion_type, void *add_input, float *add_scale);

template void NaiveConv<bfp16_t, float, float, bfp16_t>(void *input_ptr, void *output_ptr, void *weight_ptr, void *bias,
                                                        DimsVector dims_input, DimsVector dims_output, int stride_y,
                                                        int stride_x, int kernel_size_y, int kernel_size_x, int pad_y,
                                                        int pad_x, int group, int dilation, int activation_type,
                                                        float *weight_scale, int weight_scale_len,  int8_t *relu6_max,
                                                        int relu6_max_len, int fusion_type, void *add_input,
                                                        float *add_scale);

/*
 * 3d convolution funtion
 * input & output data_format is NCDHW
 * weight data_format is K-C/group-KD-KH-KW
 * depthwise is supported
 */
template <typename Tin, typename Tw, typename Tacc, typename Tout>
void NaiveConv3D(void *input_ptr, void *output_ptr, void *weight_ptr, void *bias, DimsVector dims_input,
               DimsVector dims_output, int stride_d, int stride_y, int stride_x,
               int kernel_size_d, int kernel_size_y, int kernel_size_x,
               int pad_d, int pad_y, int pad_x, int group,
               int dilation_d, int dilation_y, int dilation_x,
               int activation_type, float *scale, int scale_len,
               int fusion_type, void *add_input, float *add_scale) {
    Tin *input_data               = static_cast<Tin *>(input_ptr);
    Tw *weight_data               = static_cast<Tw *>(weight_ptr);
    Tout *output_data             = static_cast<Tout *>(output_ptr);
    Tacc *bias_data               = static_cast<Tacc *>(bias);
    int number                    = dims_output[0];
    int output_channel            = dims_output[1];
    int output_depth              = dims_output[2];
    int output_height             = dims_output[3];
    int output_width              = dims_output[4];
    int input_channel             = dims_input[1];
    int input_depth               = dims_input[2];
    int input_height              = dims_input[3];
    int input_width               = dims_input[4];
    int output_channels_per_group = output_channel / group;
    int input_channels_per_group  = input_channel / group;

    OMP_PARALLEL_FOR_
    for (int n = 0; n < number; ++n) {
        for (int g = 0; g < group; ++g) {
            int output_c_start = g * output_channels_per_group;
            int output_c_end   = (g + 1) * output_channels_per_group;
            int input_c_start  = g * input_channels_per_group;
            int input_c_end    = (g + 1) * input_channels_per_group;
            int weights_start =
                g * output_channels_per_group * input_channels_per_group * kernel_size_x * kernel_size_y * kernel_size_d;
            for (int output_c = output_c_start; output_c < output_c_end; ++output_c) {
                for (int d = 0; d < output_depth; ++d) {
                    int input_d_start = d * stride_d - pad_d;
                    for (int h = 0; h < output_height; ++h) {
                        int input_h_start = h * stride_y - pad_y;
                        for (int w = 0; w < output_width; ++w) {
                            int input_w_start = w * stride_x - pad_x;
                            Tacc result       = static_cast<Tacc>(0.0f);
                            for (int input_c = input_c_start; input_c < input_c_end; ++input_c) {
                                for (int kernel_d = 0; kernel_d < kernel_size_d; ++kernel_d) {
                                    int input_d = input_d_start + kernel_d * dilation_d;
                                    if (input_d < 0 || input_d >= input_depth) {
                                        continue;
                                    }
                                    for (int kernel_h = 0; kernel_h < kernel_size_y; ++kernel_h) {
                                        int input_h = input_h_start + kernel_h * dilation_y;
                                        if (input_h < 0 || input_h >= input_height) {
                                            continue;
                                        }
                                        for (int kernel_w = 0; kernel_w < kernel_size_x; ++kernel_w) {
                                            int input_w = input_w_start + kernel_w * dilation_x;
                                            if (input_w < 0 || input_w >= input_width) {
                                                continue;
                                            }
                                            int input_position =
                                                (((n * input_channel + input_c) * input_depth + input_d) *
                                                     input_height +
                                                 input_h) *
                                                    input_width +
                                                input_w;
                                            int weight_position =
                                                weights_start +
                                                ((((output_c - output_c_start) * input_channels_per_group + input_c -
                                                   input_c_start) *
                                                      kernel_size_d +
                                                  kernel_d) *
                                                     kernel_size_y +
                                                 kernel_h) *
                                                    kernel_size_x +
                                                kernel_w;
                                            result += input_data[input_position] * weight_data[weight_position];
                                        }
                                    }
                                }
                            }

                            int output_position =
                                (((n * output_channel + output_c) * output_depth + d) * output_height + h) * output_width + w;
                            if (bias_data) {
                                result += bias_data[output_c];
                            }
                            if (sizeof(Tin) > 1) {  // float
                                FloatActivate(result, activation_type);
                                output_data[output_position] = result;
                            } else {
                                int scaleidx = scale_len == 1 ? 0 : output_c;
                                float val    = result * scale[scaleidx];
                                if (fusion_type == FusionType_Conv_Add_Activation) {
                                    val += static_cast<Tin *>(add_input)[output_position] * add_scale[output_c];
                                }
                                if (activation_type == ActivationType_ReLU) {
                                    val = std::max(0.0f, val);
                                }
                                if (fusion_type == FusionType_Conv_Activation_Add) {
                                    val += static_cast<Tin *>(add_input)[output_position] * add_scale[output_c];
                                }
                                output_data[output_position] = float2int8(val);
                            }
                        }
                    }
                }
            }
        }
    }
}

template void NaiveConv3D<float, float, float, float>(void *input_ptr, void *output_ptr, void *weight_ptr, void *bias, DimsVector dims_input,
                                                        DimsVector dims_output, int stride_d, int stride_y, int stride_x,
                                                        int kernel_size_d, int kernel_size_y, int kernel_size_x,
                                                        int pad_d, int pad_y, int pad_x, int group,
                                                        int dilation_d, int dilation_y, int dilation_x,
                                                        int activation_type, float *scale, int scale_len,
                                                        int fusion_type, void *add_input, float *add_scale);

template void NaiveConv3D<int8_t, int8_t, int32_t, int8_t>(void *input_ptr, void *output_ptr, void *weight_ptr, void *bias, DimsVector dims_input,
                                                        DimsVector dims_output, int stride_d, int stride_y, int stride_x,
                                                        int kernel_size_d, int kernel_size_y, int kernel_size_x,
                                                        int pad_d, int pad_y, int pad_x, int group,
                                                        int dilation_d, int dilation_y, int dilation_x,
                                                        int activation_type, float *scale, int scale_len,
                                                        int fusion_type, void *add_input, float *add_scale);

template void NaiveConv3D<bfp16_t, float, float, bfp16_t>(void *input_ptr, void *output_ptr, void *weight_ptr, void *bias, DimsVector dims_input,
                                                        DimsVector dims_output, int stride_d, int stride_y, int stride_x,
                                                        int kernel_size_d, int kernel_size_y, int kernel_size_x,
                                                        int pad_d, int pad_y, int pad_x, int group,
                                                        int dilation_d, int dilation_y, int dilation_x,
                                                        int activation_type, float *scale, int scale_len,
                                                        int fusion_type, void *add_input, float *add_scale);

template <typename T>
void NaivePermute(const int count, DimsVector dims, T *bottom_data, const std::vector<int> &permute_order,
                const std::vector<int> &old_steps, const std::vector<int> &new_steps, const int num_axes,
                T *top_data) {
    for (int i = 0; i < count; ++i) {
        int old_idx = 0;
        int idx     = i;
        for (int j = num_axes-1; j >= 0; --j) {
            int order = permute_order[j];
            old_idx += (idx % dims[j]) * old_steps[order];
            idx  /= dims[j];
        }
        top_data[i] = bottom_data[old_idx];
    }
};
template void NaivePermute(const int count, DimsVector dims, float *bottom_data, const std::vector<int> &permute_order,
                        const std::vector<int> &old_steps, const std::vector<int> &new_steps, const int num_axes,
                        float *top_data);

template void NaivePermute(const int count, DimsVector dims, int8_t *bottom_data, const std::vector<int> &permute_order,
                        const std::vector<int> &old_steps, const std::vector<int> &new_steps, const int num_axes,
                        int8_t *top_data);

template void NaivePermute(const int count, DimsVector dims, fp16_t *bottom_data, const std::vector<int> &permute_order,
                        const std::vector<int> &old_steps, const std::vector<int> &new_steps, const int num_axes,
                        fp16_t *top_data);

void NaiveReorg(float *bottom_data, int width, int height, int channel, int number, int stride, int forward, int mode,
                float *top_data) {
    int in_index, c2, offset, h2, w2, out_index;
    int out_c = channel / (stride * stride);
    for (int n = 0; n < number; ++n) {
        for (int c = 0; c < channel; ++c) {
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    if (mode == 0) {
                        // DCR mode
                        in_index  = w + width * (h + height * (c + channel * n));
                        c2        = c % out_c;
                        offset    = c / out_c;
                        h2        = h * stride + offset / stride;
                        w2        = w * stride + offset % stride;
                        out_index = w2 + width * stride * (h2 + height * stride * (c2 + out_c * n));
                    } else if (mode == 1) {
                        // CRD mode
                        in_index  = w + width * (h + height * (c + channel * n));
                        c2        = c / (stride * stride);
                        offset    = c % (stride * stride);
                        h2        = h * stride + offset / stride;
                        w2        = w * stride + offset % stride;
                        out_index = w2 + width * stride * (h2 + height * stride * (c2 + out_c * n));
                    } else {
                        LOGE("Naive Reorg do not support mode\n");
                        assert(-1);
                    }
                    if (forward) {
                        top_data[out_index] = bottom_data[in_index];
                    } else {
                        top_data[in_index] = bottom_data[out_index];
                    };
                }
            }
        }
    }
}

void priorbox_set_value(const int N, const float alpha, float *Y) {
    if (alpha == 0) {
        memset(Y, 0, sizeof(float) * N);  // NOLINT(caffe/alt_fn)
        return;
    }
    for (int i = 0; i < N; ++i) {
        Y[i] = alpha;
    }
}

void NaivePriorbox(PriorBoxLayerParam *param, int output_h, int output_w, float *output_data, int layer_height,
                   int layer_width, int img_height, int img_width, float step_h, float step_w) {
    int num_priors = output_h / (layer_height * layer_width * 4);

    float offset = param->offset;
    int dim      = output_h;
    int idx      = 0;
    for (int h = 0; h < layer_height; ++h) {
        for (int w = 0; w < layer_width; ++w) {
            float center_x = (w + offset) * step_w;
            float center_y = (h + offset) * step_h;
            float box_width, box_height;

            for (int s = 0; s < param->min_sizes.size(); ++s) {
                int min_size = int(param->min_sizes[s]);
                box_width = box_height = float(min_size);
                // xmin
                output_data[idx++] = (center_x - box_width / 2.) / img_width;
                // ymin
                output_data[idx++] = ((center_y - box_height / 2.) / img_height);
                // xmax
                output_data[idx++] = ((center_x + box_width / 2.) / img_width);
                // ymax
                output_data[idx++] = ((center_y + box_height / 2.) / img_height);
                // if we have max_size
                if (param->max_sizes.size() > 0) {
                    int max_size = int(param->max_sizes[s]);
                    // second prior: aspect_ratio = 1, size = sqrt(min_size *
                    // max_size)
                    box_width = box_height = (sqrt(min_size * max_size));
                    // xmin
                    output_data[idx++] = ((center_x - box_width / 2.) / img_width);
                    // ymin
                    output_data[idx++] = ((center_y - box_height / 2.) / img_height);
                    // xmax
                    output_data[idx++] = ((center_x + box_width / 2.) / img_width);
                    // ymax
                    output_data[idx++] = ((center_y + box_height / 2.) / img_height);
                }
                for (int r = 0; r < param->aspect_ratios.size(); ++r) {
                    float ar = param->aspect_ratios[r];
                    if (fabs(ar - 1.) < 1e-6) {
                        continue;
                    }
                    box_width  = min_size * sqrt(ar);
                    box_height = min_size / sqrt(ar);
                    // xmin
                    output_data[idx++] = ((center_x - box_width / 2.) / img_width);
                    // ymin
                    output_data[idx++] = ((center_y - box_height / 2.) / img_height);
                    // xmax
                    output_data[idx++] = ((center_x + box_width / 2.) / img_width);
                    // ymax
                    output_data[idx++] = ((center_y + box_height / 2.) / img_height);
                }
            }
        }
    }
    // clip默认值是false，是否进行越界处理
    // clip the prior's coordiate such that it is within [0, 1]
    if (param->clip) {
        //  归一化
        for (int d = 0; d < dim; ++d) {
            output_data[d] = std::min<float>(std::max<float>(output_data[d], (0.)), (1.));
        }
    }
    // 前面提到过，输出c维大小是2，第一部分存放预选框数据，第二部分存放variance
    // set the variance.
    float *variance_addr = output_data + output_h * output_w;
    if (param->variances.size() == 1) {
        priorbox_set_value(dim, float(param->variances[0]), variance_addr);
    } else {
        int count = 0;
        for (int h = 0; h < layer_height; ++h) {
            for (int w = 0; w < layer_width; ++w) {
                for (int i = 0; i < num_priors; ++i) {
                    for (int j = 0; j < 4; ++j) {
                        variance_addr[count] = (param->variances[j]);
                        ++count;
                    }
                }
            }
        }
    }
}

inline CodeType GetCodeType(const int number) {
    ASSERT(number > 0 && number < 4);

    switch (number) {
        case 1: {
            return PriorBoxParameter_CodeType_CORNER;
        }
        case 2: {
            return PriorBoxParameter_CodeType_CENTER_SIZE;
        }
        default: {
            return PriorBoxParameter_CodeType_CORNER_SIZE;
        }
    }
}

void DealOutput(Blob *output_blob, const int num_kept, const int num,
                std::vector<std::map<int, std::vector<float>>> &all_conf_scores,
                std::vector<LabelBBox> &all_decode_bboxes, std::vector<std::map<int, std::vector<int>>> &all_indices,
                DetectionOutputLayerParam *param) {
    float *top_data = static_cast<float *>(output_blob->GetHandle().base);
    // clear all output to 0
    priorbox_set_value(DimsVectorUtils::Count(output_blob->GetBlobDesc().dims), 0, top_data);

    // if no detection
    if (num_kept == 0) {
        LOGD("%s:Couldn't find any detections.", __FUNCTION__);
        output_blob->GetBlobDesc().dims[2] = num;
        priorbox_set_value(DimsVectorUtils::Count(output_blob->GetBlobDesc().dims), -1, top_data);

        // Generate fake results per image.
        float *top_data_tmp = top_data;
        for (int i = 0; i < num; ++i) {
            top_data_tmp[0] = static_cast<float>(i);
            top_data_tmp += 7;
        }
    } else {
        output_blob->GetBlobDesc().dims[2] = num_kept;
    }

    int count = 0;
    for (int i = 0; i < num; ++i) {
        const std::map<int, std::vector<float>> &conf_scores = all_conf_scores[i];
        const LabelBBox &decode_bboxes                       = all_decode_bboxes[i];
        for (std::map<int, std::vector<int>>::iterator it = all_indices[i].begin(); it != all_indices[i].end(); ++it) {
            int label = it->first;
            if (conf_scores.find(label) == conf_scores.end()) {
                // Something bad happened if there are no predictions for
                // current label.
                LOGE("Could not find confidence predictions for ");
                continue;
            }
            const std::vector<float> &scores = conf_scores.find(label)->second;
            int loc_label                    = param->share_location ? -1 : label;
            if (decode_bboxes.find(loc_label) == decode_bboxes.end()) {
                // Something bad happened if there are no predictions for
                // current label.
                LOGE("Could not find location predictions for ");
                continue;
            }
            const std::vector<NormalizedBBox> &bboxes = decode_bboxes.find(loc_label)->second;
            std::vector<int> &indices                 = it->second;

            for (size_t j = 0; j < indices.size(); ++j) {
                int idx                    = indices[j];
                top_data[count * 7]        = static_cast<float>(i);
                top_data[count * 7 + 1]    = static_cast<float>(label);
                top_data[count * 7 + 2]    = scores[idx];
                const NormalizedBBox &bbox = bboxes[idx];
                top_data[count * 7 + 3]    = bbox.xmin();
                top_data[count * 7 + 4]    = bbox.ymin();
                top_data[count * 7 + 5]    = bbox.xmax();
                top_data[count * 7 + 6]    = bbox.ymax();
                ++count;
            }
        }
    }
}

void NaiveDetectionOutput(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs,
                          DetectionOutputLayerParam *param) {
    ASSERT(inputs.size() >= 3);
    Blob *loc_blob   = inputs[0];
    Blob *conf_blob  = inputs[1];
    Blob *prior_blob = inputs[2];
    auto loc_dims    = loc_blob->GetBlobDesc().dims;
    auto conf_dims   = conf_blob->GetBlobDesc().dims;
    auto prior_dims  = prior_blob->GetBlobDesc().dims;
    LOGD("the loc_lob: (%d, %d, %d, %d)\n", loc_dims[0], loc_dims[1], loc_dims[2], loc_dims[3]);
    LOGD("the conf_lob: (%d, %d, %d, %d)\n", conf_dims[0], conf_dims[1], conf_dims[2], conf_dims[3]);
    LOGD("the prior_lob: (%d, %d, %d, %d)\n", prior_dims[0], prior_dims[1], prior_dims[2], prior_dims[3]);
    const int num = loc_blob->GetBlobDesc().dims[0];
    // get output blob
    Blob *output_blob = outputs[0];
    // why defination objectness_score_ ?
    const float objectness_score_ = 0.1f;

    const float *loc_data   = static_cast<const float *>(loc_blob->GetHandle().base);
    const float *conf_data  = static_cast<const float *>(conf_blob->GetHandle().base);
    const float *prior_data = static_cast<const float *>(prior_blob->GetHandle().base);

    const float *arm_conf_data = nullptr;
    const float *arm_loc_data  = nullptr;

    vector<LabelBBox> all_arm_loc_preds;

    int num_loc_classes = param->share_location ? 1 : param->num_classes;
    int num_priors      = prior_blob->GetBlobDesc().dims[2] / 4;

    // TODO: differ
    if (inputs.size() >= 4) {
        arm_conf_data = static_cast<float *>(inputs[3]->GetHandle().base);
    }
    if (inputs.size() >= 5) {
        arm_loc_data = static_cast<float *>(inputs[4]->GetHandle().base);
        GetLocPredictions(arm_loc_data, num, num_priors, num_loc_classes, param->share_location, &all_arm_loc_preds);
    }

    // Retrieve all location predictions.
    std::vector<LabelBBox> all_loc_preds;
    GetLocPredictions(loc_data, num, num_priors, num_loc_classes, param->share_location, &all_loc_preds);

    // Retrieve all confidences.
    std::vector<std::map<int, std::vector<float>>> all_conf_scores;
    if (arm_conf_data != nullptr) {
        OSGetConfidenceScores(conf_data, arm_conf_data, num, num_priors, param->num_classes, &all_conf_scores,
                              objectness_score_);
    } else {
        GetConfidenceScores(conf_data, num, num_priors, param->num_classes, &all_conf_scores);
    }

    // Retrieve all prior bboxes. It is same within a batch since we assume all
    // images in a batch are of same dimension.
    std::vector<NormalizedBBox> prior_bboxes;
    std::vector<std::vector<float>> prior_variances;
    // TODO: differ
    GetPriorBBoxes(prior_data, num_priors, &prior_bboxes, &prior_variances);

    // Decode all loc predictions to bboxes.
    std::vector<LabelBBox> all_decode_bboxes;
    const bool clip_bbox = false;
    CodeType code_type   = GetCodeType(param->code_type);
    // TODO: differ
    if (inputs.size() >= 5) {
        CasRegDecodeBBoxesAll(all_loc_preds, prior_bboxes, prior_variances, num, param->share_location, num_loc_classes,
                              param->background_label_id, code_type, param->variance_encoded_in_target, clip_bbox,
                              &all_decode_bboxes, all_arm_loc_preds);
    } else {
        DecodeBBoxesAll(all_loc_preds, prior_bboxes, prior_variances, num, param->share_location, num_loc_classes,
                        param->background_label_id, code_type, param->variance_encoded_in_target, clip_bbox,
                        &all_decode_bboxes);
    }

    int num_kept = 0;
    std::vector<std::map<int, std::vector<int>>> all_indices;
    for (int i = 0; i < num; ++i) {
        const LabelBBox &decode_bboxes                       = all_decode_bboxes[i];
        const std::map<int, std::vector<float>> &conf_scores = all_conf_scores[i];
        std::map<int, std::vector<int>> indices;
        int num_det = 0;
        for (int c = 0; c < param->num_classes; ++c) {
            if (c == param->background_label_id) {
                // Ignore background class.
                continue;
            }
            if (conf_scores.find(c) == conf_scores.end()) {
                // Something bad happened if there are no predictions for
                // current label.
                LOGE("Could not find confidence predictions for label ");
                // assert(false); // "Could not find confidence predictions for
                // label
            }
            const std::vector<float> &scores = conf_scores.find(c)->second;
            int label                        = param->share_location ? -1 : c;
            if (decode_bboxes.find(label) == decode_bboxes.end()) {
                // Something bad happened if there are no predictions for
                LOGE("Could not find location predictions for label");
                continue;
            }
            const std::vector<NormalizedBBox> &bboxes = decode_bboxes.find(label)->second;
            ApplyNMSFast(bboxes, scores, param->confidence_threshold, param->nms_param.nms_threshold, param->eta,
                         param->nms_param.top_k, &(indices[c]));
            num_det += static_cast<int>(indices[c].size());
        }
        if (param->keep_top_k > -1 && num_det > param->keep_top_k) {
            std::vector<std::pair<float, std::pair<int, int>>> score_index_pairs;
            for (std::map<int, std::vector<int>>::iterator it = indices.begin(); it != indices.end(); ++it) {
                int label                             = it->first;
                const std::vector<int> &label_indices = it->second;
                if (conf_scores.find(label) == conf_scores.end()) {
                    // Something bad happened for current label.
                    LOGE("Could not find location predictions for ");
                    continue;
                }
                const std::vector<float> &scores = conf_scores.find(label)->second;
                for (size_t j = 0; j < label_indices.size(); ++j) {
                    size_t idx = label_indices[j];
                    assert(idx < scores.size());
                    score_index_pairs.push_back(std::make_pair(scores[idx], std::make_pair(label, idx)));
                }
            }
            // Keep top k results per image.
            std::sort(score_index_pairs.begin(), score_index_pairs.end(), SortScorePairDescend<std::pair<int, int>>);
            score_index_pairs.resize(param->keep_top_k);
            // Store the new indices.
            std::map<int, std::vector<int>> new_indices;
            for (size_t j = 0; j < score_index_pairs.size(); ++j) {
                int label = score_index_pairs[j].second.first;
                int idx   = score_index_pairs[j].second.second;
                new_indices[label].push_back(idx);
            }
            all_indices.push_back(new_indices);
            num_kept += param->keep_top_k;
        } else {
            all_indices.push_back(indices);
            num_kept += num_det;
        }
    }

    DealOutput(output_blob, num_kept, num, all_conf_scores, all_decode_bboxes, all_indices, param);
}

inline void MaxMin(float lhs, float rhs, float& min, float& max) {
  if (lhs >= rhs) {
    min = rhs;
    max = lhs;
  } else {
    min = lhs;
    max = rhs;
  }
}

inline bool SuppressByIOU(const float* boxes_data, int64_t box_index1, int64_t box_index2,
                          int64_t center_point_box, float iou_threshold) {
  float x1_min{};
  float y1_min{};
  float x1_max{};
  float y1_max{};
  float x2_min{};
  float y2_min{};
  float x2_max{};
  float y2_max{};
  float intersection_x_min{};
  float intersection_x_max{};
  float intersection_y_min{};
  float intersection_y_max{};

  const float* box1 = boxes_data + 4 * box_index1;
  const float* box2 = boxes_data + 4 * box_index2;
  // center_point_box_ only support 0 or 1
  if (0 == center_point_box) {
    // boxes data format [y1, x1, y2, x2],
    MaxMin(box1[1], box1[3], x1_min, x1_max);
    MaxMin(box2[1], box2[3], x2_min, x2_max);

    intersection_x_min = std::max(x1_min, x2_min);
    intersection_x_max = std::min(x1_max, x2_max);
    if (intersection_x_max <= intersection_x_min)
      return false;

    MaxMin(box1[0], box1[2], y1_min, y1_max);
    MaxMin(box2[0], box2[2], y2_min, y2_max);
    intersection_y_min = std::max(y1_min, y2_min);
    intersection_y_max = std::min(y1_max, y2_max);
    if (intersection_y_max <= intersection_y_min)
      return false;
  } else {
    // 1 == center_point_box_ => boxes data format [x_center, y_center, width, height]
    float box1_width_half = box1[2] / 2;
    float box1_height_half = box1[3] / 2;
    float box2_width_half = box2[2] / 2;
    float box2_height_half = box2[3] / 2;

    x1_min = box1[0] - box1_width_half;
    x1_max = box1[0] + box1_width_half;
    x2_min = box2[0] - box2_width_half;
    x2_max = box2[0] + box2_width_half;

    intersection_x_min = std::max(x1_min, x2_min);
    intersection_x_max = std::min(x1_max, x2_max);
    if (intersection_x_max <= intersection_x_min)
      return false;

    y1_min = box1[1] - box1_height_half;
    y1_max = box1[1] + box1_height_half;
    y2_min = box2[1] - box2_height_half;
    y2_max = box2[1] + box2_height_half;

    intersection_y_min = std::max(y1_min, y2_min);
    intersection_y_max = std::min(y1_max, y2_max);
    if (intersection_y_max <= intersection_y_min)
      return false;
  }

  const float intersection_area = (intersection_x_max - intersection_x_min) *
                                  (intersection_y_max - intersection_y_min);

  if (intersection_area <= .0f) {
    return false;
  }

  const float area1 = (x1_max - x1_min) * (y1_max - y1_min);
  const float area2 = (x2_max - x2_min) * (y2_max - y2_min);
  const float union_area = area1 + area2 - intersection_area;

  if (area1 <= .0f || area2 <= .0f || union_area <= .0f) {
    return false;
  }

  const float intersection_over_union = intersection_area / union_area;

  return intersection_over_union > iou_threshold;
}

void NaiveNonMaxSuppression(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs,
                            NonMaxSuppressionLayerParam *param) {
    ASSERT(inputs.size() >= 2);
    int center_point_box = param->center_point_box;
    int64_t max_output_boxes_per_class = param->max_output_boxes_per_class;
    float iou_threshold = param->iou_threshold;
    float score_threshold = param->score_threshold;

    Blob *boxes_blob = inputs[0];
    Blob *scores_blob = inputs[1];
    Blob *output_blob = outputs[0];
    auto boxes_dims = boxes_blob->GetBlobDesc().dims;
    auto scores_dims = scores_blob->GetBlobDesc().dims;

    if (0 == max_output_boxes_per_class) {
        output_blob->GetBlobDesc().dims = {0, 3};
        return;
    }

    const float *boxes_data   = static_cast<const float *>(boxes_blob->GetHandle().base);
    const float *scores_data  = static_cast<const float *>(scores_blob->GetHandle().base);

    struct BoxInfoPtr {
        float score_{};
        int index_{};

        BoxInfoPtr() = default;
        explicit BoxInfoPtr(float score, int idx) : score_(score), index_(idx) {}
        inline bool operator<(const BoxInfoPtr& rhs) const {
        return score_ < rhs.score_ || (score_ == rhs.score_ && index_ > rhs.index_);
        }
    };

    struct SelectedIndex {
        SelectedIndex(int batch_index, int class_index, int box_index)
            : batch_index_(batch_index), class_index_(class_index), box_index_(box_index) {}
        SelectedIndex() = default;
        int batch_index_ = 0;
        int class_index_ = 0;
        int box_index_ = 0;
    };

    int num_batches = boxes_dims[0];
    int num_boxes = boxes_dims[1];
    int num_classes = scores_dims[1];
    std::vector<SelectedIndex> selected_indices;
    std::vector<BoxInfoPtr> selected_boxes_inside_class;
    selected_boxes_inside_class.reserve(std::min<size_t>(static_cast<size_t>(max_output_boxes_per_class), num_boxes));

    for (int batch_index = 0; batch_index < num_batches; ++batch_index) {
        for (int class_index = 0; class_index < num_classes; ++class_index) {
            int box_score_offset = (batch_index * num_classes + class_index) * num_boxes;
            const float* batch_boxes = boxes_data + (batch_index * num_boxes * 4);
            std::vector<BoxInfoPtr> candidate_boxes;
            candidate_boxes.reserve(num_boxes);

            // Filter by score_threshold_
            const auto* class_scores = scores_data + box_score_offset;
            for (int box_index = 0; box_index < num_boxes; ++box_index, ++class_scores) {
                if (*class_scores > score_threshold) {
                    candidate_boxes.emplace_back(*class_scores, box_index);
                }
            }
            std::priority_queue<BoxInfoPtr, std::vector<BoxInfoPtr>> sorted_boxes(
                std::less<BoxInfoPtr>(), std::move(candidate_boxes));

            selected_boxes_inside_class.clear();
            // Get the next box with top score, filter by iou_threshold
            while (!sorted_boxes.empty() && static_cast<int>(selected_boxes_inside_class.size())
                    < max_output_boxes_per_class) {
                const BoxInfoPtr& next_top_score = sorted_boxes.top();

                bool selected = true;
                // Check with existing selected boxes for this class, suppress if exceed the IOU (Intersection Over Union) threshold
                for (const auto& selected_index : selected_boxes_inside_class) {
                    if (SuppressByIOU(batch_boxes, next_top_score.index_, selected_index.index_,
                                      center_point_box, iou_threshold)) {
                        selected = false;
                        break;
                    }
                }

                if (selected) {
                    selected_boxes_inside_class.push_back(next_top_score);
                    selected_indices.emplace_back(batch_index, class_index, next_top_score.index_);
                }
                sorted_boxes.pop();
            }
        }
    }

    const auto last_dim = 3;
    const auto num_selected = selected_indices.size();
    int *output_blob_data = static_cast<int *>(output_blob->GetHandle().base);
    output_blob->GetBlobDesc().dims = {static_cast<int>(num_selected), last_dim};

    ASSERT(last_dim * sizeof(int) == sizeof(SelectedIndex));
    memcpy(output_blob_data, selected_indices.data(), num_selected * sizeof(SelectedIndex));
}

void NaiveColorToGray(const uint8_t *src, uint8_t *dst, int h, int w, int channel, bool bgr_order) {
    int offset = 0;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            unsigned c1      = src[offset * channel + 0];
            unsigned c2      = src[offset * channel + 1];
            unsigned c3      = src[offset * channel + 2];
            unsigned b       = bgr_order ? c1 : c3;
            unsigned g       = c2;
            unsigned r       = bgr_order ? c3 : c1;
            float gray_color = 0.114f * b + 0.587 * g + 0.299 * r;
            dst[offset]      = gray_color;
            offset += 1;
        }
    }
}

void NaiveBGROrBGRAToGray(const uint8_t *src, uint8_t *dst, int h, int w, int channel) {
    return NaiveColorToGray(src, dst, h, w, channel, true);
}

void NaiveRGBOrRGBAToGray(const uint8_t *src, uint8_t *dst, int h, int w, int channel) {
    return NaiveColorToGray(src, dst, h, w, channel, false);
}

void NaiveYUVToBGROrBGRALoop(const unsigned char *yptr0, const unsigned char *yptr1, const unsigned char *vuptr,
                             unsigned char *rgb0, unsigned char *rgb1, int remain, bool is_nv12, int channel) {
    for (; remain > 0; remain -= 2) {
        int u, v;
        if (is_nv12) {
            u = (vuptr[0] > 240 ? 240 : vuptr[0]) - 128;
            v = (vuptr[1] > 240 ? 240 : vuptr[1]) - 128;
        } else {
            v = (vuptr[0] > 240 ? 240 : vuptr[0]) - 128;
            u = (vuptr[1] > 240 ? 240 : vuptr[1]) - 128;
        }

        int ruv = 102 * v;
        int guv = -52 * v + -25 * u;
        int buv = 129 * u;

#define SATURATE_CAST_UCHAR(X) (unsigned char)std::min(std::max(X, 0), 255);

        int y00 = yptr0[0] * 74 - 1135;
        if (channel == 4)
            rgb0[3] = 255;
        rgb0[0 * channel + 2] = SATURATE_CAST_UCHAR((y00 + ruv) >> 6);
        rgb0[0 * channel + 1] = SATURATE_CAST_UCHAR((y00 + guv) >> 6);
        rgb0[0 * channel + 0] = SATURATE_CAST_UCHAR((y00 + buv) >> 6);

        int y01 = yptr0[1] * 74 - 1135;
        if (channel == 4)
            rgb0[7] = 255;
        rgb0[1 * channel + 2] = SATURATE_CAST_UCHAR((y01 + ruv) >> 6);
        rgb0[1 * channel + 1] = SATURATE_CAST_UCHAR((y01 + guv) >> 6);
        rgb0[1 * channel + 0] = SATURATE_CAST_UCHAR((y01 + buv) >> 6);

        int y10 = yptr1[0] * 74 - 1135;
        if (channel == 4)
            rgb1[3] = 255;
        rgb1[0 * channel + 2] = SATURATE_CAST_UCHAR((y10 + ruv) >> 6);
        rgb1[0 * channel + 1] = SATURATE_CAST_UCHAR((y10 + guv) >> 6);
        rgb1[0 * channel + 0] = SATURATE_CAST_UCHAR((y10 + buv) >> 6);

        int y11 = yptr1[1] * 74 - 1135;
        if (channel == 4)
            rgb1[7] = 255;
        rgb1[1 * channel + 2] = SATURATE_CAST_UCHAR((y11 + ruv) >> 6);
        rgb1[1 * channel + 1] = SATURATE_CAST_UCHAR((y11 + guv) >> 6);
        rgb1[1 * channel + 0] = SATURATE_CAST_UCHAR((y11 + buv) >> 6);

#undef SATURATE_CAST_UCHAR

        yptr0 += 2;
        yptr1 += 2;
        vuptr += 2;
        rgb0 += 2 * channel;
        rgb1 += 2 * channel;
    }
}

void NaiveYUVToBGROrBGRA(const unsigned char *yuv, unsigned char *bgr, const int channel, const int h, const int w,
                         bool is_nv12) {
    const unsigned char *yptr  = yuv;
    const unsigned char *vuptr = yuv + w * h;

    for (int y = 0; y < h; y += 2) {
        const unsigned char *yptr0 = yptr;
        const unsigned char *yptr1 = yptr + w;
        unsigned char *rgb0        = bgr;
        unsigned char *rgb1        = bgr + w * channel;

        NaiveYUVToBGROrBGRALoop(yptr0, yptr1, vuptr, rgb0, rgb1, w, is_nv12, channel);

        yptr += 2 * w;
        vuptr += w;
        bgr += 2 * channel * w;
    }
}

void NaiveDequant(const int8_t *input_ptr, const float *scale_ptr, int scale_len, float *output, DimsVector dims) {
    int batch   = DimsFunctionUtils::GetDim(dims, 0);
    int channel = DimsFunctionUtils::GetDim(dims, 1);
    int hw_size = DimsVectorUtils::Count(dims, 2);
    for (int n = 0; n < batch; n++) {
        OMP_PARALLEL_FOR_
        for (int c = 0; c < channel; c++) {
            int offset    = n * channel * hw_size + c * hw_size;
            int scale_idx = scale_len == 1 ? 0 : c;
            for (int hw = 0; hw < hw_size; hw++) {
                output[offset + hw] = scale_ptr[scale_idx] * static_cast<float>(input_ptr[offset + hw]);
            }
        }
    }
}

void NaiveQuant(const float *input_ptr, const float *scale_ptr, int scale_len, int8_t *output, DimsVector dims) {
    int batch   = DimsFunctionUtils::GetDim(dims, 0);
    int channel = DimsFunctionUtils::GetDim(dims, 1);
    int hw_size = DimsVectorUtils::Count(dims, 2);
    for (int n = 0; n < batch; n++) {
        OMP_PARALLEL_FOR_
        for (int c = 0; c < channel; c++) {
            int offset    = n * channel * hw_size + c * hw_size;
            int scale_idx = scale_len == 1 ? 0 : c;
            for (int hw = 0; hw < hw_size; hw++) {
                if (scale_ptr[scale_idx] != 0)
                    output[offset + hw] = float2int8(input_ptr[offset + hw] / scale_ptr[scale_idx]);
                else
                    output[offset + hw] = 0;
            }
        }
    }
}
void NaiveDequantBias(const int8_t *input_ptr, const float *scale_ptr, const int8_t *zero_point_ptr, int scale_len,
                      float *output, DimsVector dims) {
    int batch   = DimsFunctionUtils::GetDim(dims, 0);
    int channel = DimsFunctionUtils::GetDim(dims, 1);
    int hw_size = DimsVectorUtils::Count(dims, 2);
    for (int n = 0; n < batch; n++) {
        OMP_PARALLEL_FOR_
        for (int c = 0; c < channel; c++) {
            int offset    = n * channel * hw_size + c * hw_size;
            int scale_idx = scale_len == 1 ? 0 : c;
            for (int hw = 0; hw < hw_size; hw++) {
                output[offset + hw] = scale_ptr[scale_idx] * (static_cast<float>(input_ptr[offset + hw]) -
                                                              static_cast<float>(zero_point_ptr[scale_idx]));
            }
        }
    }    
}

void NaiveQuantBias(const float *input_ptr, const float *scale_ptr, const int8_t *zero_point_ptr, int scale_len,
                    int8_t *output, DimsVector dims) {
    int batch   = DimsFunctionUtils::GetDim(dims, 0);
    int channel = DimsFunctionUtils::GetDim(dims, 1);
    int hw_size = DimsVectorUtils::Count(dims, 2);
    for (int n = 0; n < batch; n++) {
        OMP_PARALLEL_FOR_
        for (int c = 0; c < channel; c++) {
            int offset    = n * channel * hw_size + c * hw_size;
            int scale_idx = scale_len == 1 ? 0 : c;
            for (int hw = 0; hw < hw_size; hw++) {
                if (scale_ptr[scale_idx] != 0) {
                    output[offset + hw] = float2int8(input_ptr[offset + hw] / scale_ptr[scale_idx] +
                                                     static_cast<float>(zero_point_ptr[scale_idx]));
              } else
                    output[offset + hw] = 0;
            }
       }
    }
}    

}  // namespace TNN_NS

                                                     
