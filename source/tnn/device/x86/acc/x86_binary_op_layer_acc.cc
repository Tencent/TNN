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

#include "tnn/device/x86/acc/x86_binary_op_layer_acc.h"
#include "tnn/device/x86/x86_common.h"
#include "tnn/device/x86/x86_context.h"
#include "tnn/utils/data_format_converter.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_utils.h"
#include "tnn/device/x86/acc/Float4.h"
#include "tnn/device/x86/acc/Float8.h"

namespace TNN_NS {

template<X86BinaryOpType type>
float binary_op(const float &a, const float &b) {
    return a;
}
template<> float binary_op<X86BinaryOpType::kADD>(const float &a, const float &b) {
    return a + b;
}
template<> float binary_op<X86BinaryOpType::kSUB>(const float &a, const float &b) {
    return a - b;
}
template<> float binary_op<X86BinaryOpType::kMUL>(const float &a, const float &b) {
    return a * b;
}
template<> float binary_op<X86BinaryOpType::kDIV>(const float &a, const float &b) {
    return a / b;
}
template<> float binary_op<X86BinaryOpType::kMAX>(const float &a, const float &b) {
    return a > b ? a : b;
}
template<> float binary_op<X86BinaryOpType::kMIN>(const float &a, const float &b) {
    return a < b ? a : b;
}

template<X86BinaryOpType type, typename VEC>
VEC binary_op(const VEC &a, const VEC &b) {
    return a;
}
template<> Float4 binary_op<X86BinaryOpType::kADD, Float4>(const Float4 &a, const Float4 &b) {
    return Float4::add(a, b);
}
template<> Float4 binary_op<X86BinaryOpType::kSUB, Float4>(const Float4 &a, const Float4 &b) {
    return Float4::sub(a, b);
}
template<> Float4 binary_op<X86BinaryOpType::kMUL, Float4>(const Float4 &a, const Float4 &b) {
    return Float4::mul(a, b);
}
template<> Float4 binary_op<X86BinaryOpType::kDIV, Float4>(const Float4 &a, const Float4 &b) {
    return Float4::div(a, b);
}
template<> Float4 binary_op<X86BinaryOpType::kMAX, Float4>(const Float4 &a, const Float4 &b) {
    return Float4::max(a, b);
}
template<> Float4 binary_op<X86BinaryOpType::kMIN, Float4>(const Float4 &a, const Float4 &b) {
    return Float4::min(a, b);
}
template<> Float8 binary_op<X86BinaryOpType::kADD, Float8>(const Float8 &a, const Float8 &b) {
    return Float8::add(a, b);
}
template<> Float8 binary_op<X86BinaryOpType::kSUB, Float8>(const Float8 &a, const Float8 &b) {
    return Float8::sub(a, b);
}
template<> Float8 binary_op<X86BinaryOpType::kMUL, Float8>(const Float8 &a, const Float8 &b) {
    return Float8::mul(a, b);
}
template<> Float8 binary_op<X86BinaryOpType::kDIV, Float8>(const Float8 &a, const Float8 &b) {
    return Float8::div(a, b);
}
template<> Float8 binary_op<X86BinaryOpType::kMAX, Float8>(const Float8 &a, const Float8 &b) {
    return Float8::max(a, b);
}
template<> Float8 binary_op<X86BinaryOpType::kMIN, Float8>(const Float8 &a, const Float8 &b) {
    return Float8::min(a, b);
}

static inline void PadShape(const int pad_size, const int dim_size, DimsVector &pad_shape, DimsVector in_shape) {
    int j = 0;
    for (; j < pad_size; j++) {
        pad_shape[j] = 1;
    }
    for (; j < dim_size; j++) {
        pad_shape[j] = in_shape[j - pad_size];
    }
}

static void BroadCastTypeFilter(const DimsVector &dims_output, const DimsVector &dims_input, BroadcastType &type) {
    if (DimsVectorUtils::Equal(dims_output, dims_input)) {
        type = BroadcastTypeNormal;
        return;
    }
    if (DimsVectorUtils::Equal(dims_output, dims_input, 1) &&
        DimsVectorUtils::Count(dims_input, 0, 1) == 1) {
        type = BroadcastTypeElement;
        return;
    }
    if (DimsVectorUtils::Equal(dims_output, dims_input, 2) &&
        DimsVectorUtils::Count(dims_input, 0, 2) == 1) {
        type = BroadcastTypeHeightWidth;
        return;
    }
    if (DimsVectorUtils::Equal(dims_output, dims_input, 3) &&
        DimsVectorUtils::Count(dims_input, 0, 3) == 1) {
        type = BroadcastTypeWidth;
        return;
    }
    int broadcast_count = DimsVectorUtils::Count(dims_input);
    if (broadcast_count == 1) {
        type = BroadcastTypeSingle;
    } else if (broadcast_count == dims_output[1]) {
        // broadcast dim = [1, channel, 1...]
        if (dims_input[1] == dims_output[1]) {
            type = BroadcastTypeChannel;
        } else {
            type = BroadcastTypeGeneral;
        }
    } else {
        type = BroadcastTypeGeneral;
    }
    return;
}

static void BroadCastInit(const DimsVector &dims, const DimsVector &dims0, const DimsVector &dims1, BroadcastType &type,
                          DimsVector &dims_broadcast, bool &swap_flag) {
    if (DimsVectorUtils::Equal(dims0, dims1)) {
        type = BroadcastTypeNormal;
        dims_broadcast.clear();
    } else if (DimsVectorUtils::Equal(dims0, dims1, 1) &&
              (DimsVectorUtils::Count(dims0, 0, 1) == 1 ||
               DimsVectorUtils::Count(dims1, 0, 1) == 1)) {
        type = BroadcastTypeElement;
        dims_broadcast.clear();
        if (dims0[0] < dims1[0])
            swap_flag = true;
    } else if (DimsVectorUtils::Equal(dims0, dims1, 2) &&
              (DimsVectorUtils::Count(dims0, 0, 2) == 1 ||
               DimsVectorUtils::Count(dims1, 0, 2) == 1)) {
        type = BroadcastTypeHeightWidth;
        dims_broadcast.clear();
        if (dims0[1] < dims1[1])
            swap_flag = true;
    } else if (DimsVectorUtils::Equal(dims0, dims1, 3) &&
              (DimsVectorUtils::Count(dims0, 0, 3) == 1 ||
               DimsVectorUtils::Count(dims1, 0, 3) == 1)) {
        type = BroadcastTypeWidth;
        dims_broadcast.clear();
        if (dims0[1] < dims1[1])
            swap_flag = true;
    } else if (DimsVectorUtils::Equal(dims0, dims)) {
        dims_broadcast = dims1;
    } else {
        dims_broadcast = dims0;
        swap_flag      = true;
    }
}

static void BinaryComputeOffset(DimsVector &offset, const DimsVector dims_in, const DimsVector dims_out) {
    DimsVector dims_pad_in;
    int pad_size = dims_out.size() - dims_in.size();
    int i = 0;
    for (; i < pad_size; i++) {
        dims_pad_in.push_back(1);
    }
    for (; i < dims_out.size(); i++) {
        dims_pad_in.push_back(dims_in[i - pad_size]);
    }

    offset.resize(dims_out.size());
    int s = 1;
    for (i = dims_out.size() - 1; i >= 0; i--) {
        offset[i] = (dims_pad_in[i] == dims_out[i]) ? s : 0;
        s *= dims_pad_in[i];
    }
}

static void BinaryComputeFirst(const DimsVector input_offset, const DimsVector output_offset,
                               const DimsVector output_shape, const float* input_ptr, float* output_ptr) {
    DimsVector out_shape;
    DimsVector in_offset;
    DimsVector ou_offset;
    // support maximum 6 dimension, may be extended in furture
    out_shape.resize(6);
    in_offset.resize(6);
    ou_offset.resize(6);
    // if dim < 6, pad to 6
    int pad_size = 6 - output_shape.size();
    for (int i = 0; i < pad_size; i++) {
        out_shape[i] = 1;
        in_offset[i] = 0;
        ou_offset[i] = 0;
    }
    for (int i = pad_size; i < 6; i++) {
        out_shape[i] = output_shape[i - pad_size];
        in_offset[i] = input_offset[i - pad_size];
        ou_offset[i] = output_offset[i - pad_size];
    }

    for (int i0 = 0; i0 < out_shape[0]; i0++) {
        auto in_i0 = input_ptr + i0 * in_offset[0];
        auto ou_i0 = output_ptr + i0 * ou_offset[0];
        for (int i1 = 0; i1 < out_shape[1]; i1++) {
            auto in_i1 = in_i0 + i1 * in_offset[1];
            auto ou_i1 = ou_i0 + i1 * ou_offset[1];
            for (int i2 = 0; i2 < out_shape[2]; i2++) {
                auto in_i2 = in_i1 + i2 * in_offset[2];
                auto ou_i2 = ou_i1 + i2 * ou_offset[2];
                for (int i3 = 0; i3 < out_shape[3]; i3++) {
                    auto in_i3 = in_i2 + i3 * in_offset[3];
                    auto ou_i3 = ou_i2 + i3 * ou_offset[3];
                    for (int i4 = 0; i4 < out_shape[4]; i4++) {
                        auto in_i4 = in_i3 + i4 * in_offset[4];
                        auto ou_i4 = ou_i3 + i4 * ou_offset[4];
                        for (int i5 = 0; i5 < out_shape[5]; i5++) {
                            auto in_i5 = in_i4 + i5 * in_offset[5];
                            auto ou_i5 = ou_i4 + i5 * ou_offset[5];
                            ou_i5[0] = in_i5[0];
                        }
                    }
                }
            }
        }
    }
}

template <X86BinaryOpType op_type>
void BinaryCompute(const DimsVector input_offset, const DimsVector output_offset,
                          const DimsVector output_shape, const float* input_ptr, float* output_ptr) {
    DimsVector out_shape;
    DimsVector in_offset;
    DimsVector ou_offset;
    // support maximum 6 dimension, may be extended in furture
    out_shape.resize(6);
    in_offset.resize(6);
    ou_offset.resize(6);
    // if dim < 6, pad to 6
    int pad_size = 6 - output_shape.size();
    for (int i = 0; i < pad_size; i++) {
        out_shape[i] = 1;
        in_offset[i] = 0;
        ou_offset[i] = 0;
    }
    for (int i = pad_size; i < 6; i++) {
        out_shape[i] = output_shape[i - pad_size];
        in_offset[i] = input_offset[i - pad_size];
        ou_offset[i] = output_offset[i - pad_size];
    }

    for (int i0 = 0; i0 < out_shape[0]; i0++) {
        auto in_i0 = input_ptr + i0 * in_offset[0];
        auto ou_i0 = output_ptr + i0 * ou_offset[0];
        for (int i1 = 0; i1 < out_shape[1]; i1++) {
            auto in_i1 = in_i0 + i1 * in_offset[1];
            auto ou_i1 = ou_i0 + i1 * ou_offset[1];
            for (int i2 = 0; i2 < out_shape[2]; i2++) {
                auto in_i2 = in_i1 + i2 * in_offset[2];
                auto ou_i2 = ou_i1 + i2 * ou_offset[2];
                for (int i3 = 0; i3 < out_shape[3]; i3++) {
                    auto in_i3 = in_i2 + i3 * in_offset[3];
                    auto ou_i3 = ou_i2 + i3 * ou_offset[3];
                    for (int i4 = 0; i4 < out_shape[4]; i4++) {
                        auto in_i4 = in_i3 + i4 * in_offset[4];
                        auto ou_i4 = ou_i3 + i4 * ou_offset[4];
                        for (int i5 = 0; i5 < out_shape[5]; i5++) {
                            auto in_i5 = in_i4 + i5 * in_offset[5];
                            auto ou_i5 = ou_i4 + i5 * ou_offset[5];
                            ou_i5[0] = binary_op<op_type>(ou_i5[0], in_i5[0]);
                        }
                    }
                }
            }
        }
    }
}

template <X86BinaryOpType op_type>
void BinaryGeneral(DimsVector output_shape, const std::vector<DimsVector> &input_shapes,
                   float *output_ptr, std::vector<float *> &input_ptrs) {
    DimsVector output_offset;
    BinaryComputeOffset(output_offset, output_shape, output_shape);

    for (int i = 0; i < input_shapes.size(); i++) {
        auto input_shape = input_shapes[i];
        float *input_ptr = input_ptrs[i];

        DimsVector input_offset;
        BinaryComputeOffset(input_offset, input_shape, output_shape);

        if (i == 0) {
            BinaryComputeFirst(input_offset, output_offset, output_shape, input_ptr, output_ptr);
        } else {
            BinaryCompute<op_type>(input_offset, output_offset, output_shape, input_ptr, output_ptr);
        }
    }
}

/*
Binary func with different opreator,
set dims0 full shape, dims1 broadcast shape, so we need to swap input ptrs
*/
template <X86BinaryOpType op_type, typename VEC, int pack>
Status BinaryFunc(float *output_ptr, const float *input0_ptr, const float *input1_ptr, DimsVector &dims0, DimsVector &dims1, DimsVector &output_dims) {
    DimsVector dims = DimsVectorUtils::Max(dims0, dims1);
    DimsVector dims_broadcast;
    BroadcastType type = BroadcastTypeUnknown;
    auto _input0       = input0_ptr;
    auto _input1       = input1_ptr;
    bool swap_flag     = false;

    BroadCastInit(dims, dims0, dims1, type, dims_broadcast, swap_flag);

    if (swap_flag) {
        std::swap(_input0, _input1);
    }

    if (dims_broadcast.size() == 1) {
        type = BroadcastTypeSingle; // dims_broadcast[0] == 1
    } else if (dims_broadcast.size() >= 2) {
        type = (dims_broadcast[1] == 1) ? BroadcastTypeSingle : BroadcastTypeChannel;
    }

    size_t count = DimsVectorUtils::Count(dims);
    size_t batch_stride = 1;
    size_t channel_stride = 1;
    if (dims.size() > 1) {
        batch_stride = DimsVectorUtils::Count(dims, 1);
    }
    if (dims.size() > 2) {
        channel_stride = DimsVectorUtils::Count(dims, 2);
    }

    if (type == BroadcastTypeNormal) {
        size_t n = 0;
        for (; n + pack - 1 < count; n += pack) {
            VEC v1 = VEC::loadu(_input0 + n);
            VEC v2 = VEC::loadu(_input1 + n);
            VEC::saveu(output_ptr + n, binary_op<op_type, VEC>(v1, v2));
        }
        for (; n < count; n++) {
            output_ptr[n] = binary_op<op_type>(_input0[n], _input1[n]);
        }
        return TNN_OK;
    }

    if (swap_flag) {
        if (type == BroadcastTypeSingle) {
            // broadcast single
            VEC v2 = VEC(_input1[0]);
            size_t n = 0;
            for (; n + pack - 1 < count; n += pack) {
                VEC v1 = VEC::loadu(_input0 + n);
                VEC::saveu(output_ptr + n, binary_op<op_type, VEC>(v2, v1));
            }
            for (; n < count; n++) {
                output_ptr[n] = binary_op<op_type>(_input1[0], _input0[n]);
            }
        } else if (type == BroadcastTypeChannel) {
            // broadcast channel
            for (int b = 0; b < dims[0]; b++) {
                for (int c = 0; c < dims[1]; c++) {
                    VEC v2 = VEC(_input1[c]);
                    auto _input0_c = _input0 + b * batch_stride + c * channel_stride;
                    auto _output_c = output_ptr + b * batch_stride + c * channel_stride;
                    int hw = 0;
                    for (; hw + pack - 1 < channel_stride; hw += pack) {
                        VEC v1 = VEC::loadu(_input0_c + hw);
                        VEC::saveu(_output_c + hw, binary_op<op_type, VEC>(v2, v1));
                    }
                    for (; hw < channel_stride; hw++) {
                        _output_c[hw] = binary_op<op_type>(_input1[c], _input0_c[hw]);
                    }
                }
            }
        } else if (type == BroadcastTypeElement) {
            // broadcast chw
            for (int b = 0; b < dims[0]; b++) {
                auto _input0_b = _input0 + b * batch_stride;
                auto _output_b = output_ptr + b * batch_stride;
                int chw = 0;
                for (; chw + pack - 1 < batch_stride; chw += pack) {
                    VEC v2 = VEC::loadu(_input1 + chw);
                    VEC v1 = VEC::loadu(_input0_b + chw);
                    VEC::saveu(_output_b + chw, binary_op<op_type, VEC>(v2, v1));
                }
                for (; chw < batch_stride; chw++) {
                    _output_b[chw] = binary_op<op_type>(_input1[chw], _input0_b[chw]);
                }
            }
        } else if (type == BroadcastTypeHeightWidth) {
            // broadcast hw
            for (int b = 0; b < dims[0]; b++) {
                for (int c = 0; c < dims[1]; c++) {
                    auto _input0_c = _input0 + b * batch_stride + c * channel_stride;
                    auto _output_c = output_ptr + b * batch_stride + c * channel_stride;
                    int hw = 0;
                    for (; hw + pack - 1 < channel_stride; hw += pack) {
                        VEC v2 = VEC::loadu(_input1 + hw);
                        VEC v1 = VEC::loadu(_input0_c + hw);
                        VEC::saveu(_output_c + hw, binary_op<op_type, VEC>(v2, v1));
                    }
                    for (; hw < channel_stride; hw++) {
                        _output_c[hw] = binary_op<op_type>(_input1[hw], _input0_c[hw]);
                    }
                }
            }
        } else if (type == BroadcastTypeWidth) {
            // broadcast w
            for (int b = 0; b < dims[0]; b++) {
                for (int c = 0; c < dims[1]; c++) {
                    auto _input0_c = _input0 + b * batch_stride + c * channel_stride;
                    auto _output_c = output_ptr + b * batch_stride + c * channel_stride;
                    for (int h = 0; h < dims[2]; h++) {
                        auto _input0_h = _input0_c + h * dims[3];
                        auto _output_h = _output_c + h * dims[3];
                        int w = 0;
                        for (; w + pack - 1 < dims[3]; w += pack) {
                            VEC v2 = VEC::loadu(_input1 + w);
                            VEC v1 = VEC::loadu(_input0_h + w);
                            VEC::saveu(_output_h + w, binary_op<op_type, VEC>(v2, v1));
                        }
                        for (; w < dims[3]; w++) {
                            _output_h[w] = binary_op<op_type>(_input1[w], _input0_h[w]);
                        }
                    }
                }
            }
        } else {
            LOGE("Error: invalid add type\n");
            return Status(TNNERR_LAYER_ERR, "Error: Binary layer's unsupported broadcast type");
        }
    } else {
        if (type == BroadcastTypeSingle) {
            // broadcast single
            VEC v2 = VEC(_input1[0]);
            size_t n = 0;
            for (; n + pack - 1 < count; n += pack) {
                VEC v1 = VEC::loadu(_input0 + n);
                VEC::saveu(output_ptr + n, binary_op<op_type, VEC>(v1, v2));
            }
            for (; n < count; n++) {
                output_ptr[n] = binary_op<op_type>(_input0[n], _input1[0]);
            }
        } else if (type == BroadcastTypeChannel) {
            // broadcast channel
            for (int b = 0; b < dims[0]; b++) {
                for (int c = 0; c < dims[1]; c++) {
                    VEC v2 = VEC(_input1[c]);
                    auto _input0_c = _input0 + b * batch_stride + c * channel_stride;
                    auto _output_c = output_ptr + b * batch_stride + c * channel_stride;
                    int hw = 0;
                    for (; hw + pack - 1 < channel_stride; hw += pack) {
                        VEC v1 = VEC::loadu(_input0_c + hw);
                        VEC::saveu(_output_c + hw, binary_op<op_type, VEC>(v1, v2));
                    }
                    for (; hw < channel_stride; hw++) {
                        _output_c[hw] = binary_op<op_type>(_input0_c[hw], _input1[c]);
                    }
                }
            }
        } else if (type == BroadcastTypeElement) {
            // broadcast chw
            for (int b = 0; b < dims[0]; b++) {
                auto _input0_b = _input0 + b * batch_stride;
                auto _output_b = output_ptr + b * batch_stride;
                int chw = 0;
                for (; chw + pack - 1 < batch_stride; chw += pack) {
                    VEC v2 = VEC::loadu(_input1 + chw);
                    VEC v1 = VEC::loadu(_input0_b + chw);
                    VEC::saveu(_output_b + chw, binary_op<op_type, VEC>(v1, v2));
                }
                for (; chw < batch_stride; chw++) {
                    _output_b[chw] = binary_op<op_type>(_input0_b[chw], _input1[chw]);
                }
            }
        } else if (type == BroadcastTypeHeightWidth) {
            // broadcast hw
            for (int b = 0; b < dims[0]; b++) {
                for (int c = 0; c < dims[1]; c++) {
                    auto _input0_c = _input0 + b * batch_stride + c * channel_stride;
                    auto _output_c = output_ptr + b * batch_stride + c * channel_stride;
                    int hw = 0;
                    for (; hw + pack - 1 < channel_stride; hw += pack) {
                        VEC v2 = VEC::loadu(_input1 + hw);
                        VEC v1 = VEC::loadu(_input0_c + hw);
                        VEC::saveu(_output_c + hw, binary_op<op_type, VEC>(v1, v2));
                    }
                    for (; hw < channel_stride; hw++) {
                        _output_c[hw] = binary_op<op_type>(_input0_c[hw], _input1[hw]);
                    }
                }
            }
        } else if (type == BroadcastTypeWidth) {
            // broadcast w
            for (int b = 0; b < dims[0]; b++) {
                for (int c = 0; c < dims[1]; c++) {
                    auto _input0_c = _input0 + b * batch_stride + c * channel_stride;
                    auto _output_c = output_ptr + b * batch_stride + c * channel_stride;
                    for (int h = 0; h < dims[2]; h++) {
                        auto _input0_h = _input0_c + h * dims[3];
                        auto _output_h = _output_c + h * dims[3];
                        int w = 0;
                        for (; w + pack - 1 < dims[3]; w += pack) {
                            VEC v2 = VEC::loadu(_input1 + w);
                            VEC v1 = VEC::loadu(_input0_h + w);
                            VEC::saveu(_output_h + w, binary_op<op_type, VEC>(v1, v2));
                        }
                        for (; w < dims[3]; w++) {
                            _output_h[w] = binary_op<op_type>(_input0_h[w], _input1[w]);
                        }
                    }
                }
            }
        } else {
            LOGE("Error: invalid add type\n");
            return Status(TNNERR_LAYER_ERR, "Error: Binary layer's unsupported broadcast type");
        }
    }

    return TNN_OK;
}

X86BinaryOpLayerAcc::~X86BinaryOpLayerAcc() {}

Status X86BinaryOpLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                             const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<MultidirBroadcastLayerParam *>(param);
    CHECK_PARAM_NULL(layer_param);
    auto layer_res = dynamic_cast<EltwiseLayerResource *>(resource);

    Status ret;
    if (inputs.size() == 1 && layer_res && layer_res->element_handle.GetDataType() == DATA_TYPE_HALF) {
        LayerResource *fp32_res = nullptr;
        LayerType layer_type    = GlobalConvertLayerType(layer_param->type);
        RETURN_ON_NEQ(ConvertHalfResource(layer_type, layer_res, &fp32_res), TNN_OK);
        binary_acc_f32_resource_ = std::shared_ptr<LayerResource>(fp32_res);
        ret                      = X86LayerAcc::Init(context, param, binary_acc_f32_resource_.get(), inputs, outputs);
    } else {
        ret = X86LayerAcc::Init(context, param, resource, inputs, outputs);
    }

    if (ret != TNN_OK) {
        return ret;
    }

    // prepare input shapes
    input_shapes_.clear();
    input_shapes_.reserve(4);
    auto output = outputs[0];
    auto output_dims = output->GetBlobDesc().dims;

    if (layer_res && inputs.size() == 1) {
        DimsVector input_shape0 = inputs[0]->GetBlobDesc().dims;
        if (layer_param->weight_input_index == 0) {
            // bias as another input
            input_shapes_.push_back(layer_res->element_shape);
            input_shapes_.push_back(input_shape0);
        } else {
            input_shapes_.push_back(input_shape0);
            input_shapes_.push_back(layer_res->element_shape);
        }
    } else {
        if (inputs.size() == 1) {
            input_shapes_.push_back(inputs[0]->GetBlobDesc().dims);
            input_shapes_.push_back(inputs[0]->GetBlobDesc().dims);
        } else {
            for (size_t inid = 0; inid < inputs.size(); inid++) {
                input_shapes_.push_back(inputs[inid]->GetBlobDesc().dims);
            }
        }
    }

    btype_ = BroadcastTypeUnknown;
    // check broadcast type is general or other optimized ncxhwx types
    // if type is general, go to nchw general impl
    DimsVector input_pad_shape;
    input_pad_shape.resize(output_dims.size());
    for (int i = 0; i < input_shapes_.size(); i++) {
        int pad_size = output_dims.size() - input_shapes_[i].size();
        PadShape(pad_size, output_dims.size(), input_pad_shape, input_shapes_[i]);
        BroadCastTypeFilter(output_dims, input_pad_shape, btype_);
        if (btype_ == BroadcastTypeGeneral) {
            break;
        }
    }

    // set binary function pointer
    binary_func_ = BinaryFunc<X86BinaryOpType::kADD, Float4, 4>;
    binary_general_func_ = BinaryGeneral<X86BinaryOpType::kADD>;

    if (arch_ == avx2) {
        switch(op_type_) {
            case X86BinaryOpType::kADD :
                binary_func_ = BinaryFunc<X86BinaryOpType::kADD, Float8, 8>;
                break;
            case X86BinaryOpType::kSUB :
                binary_func_ = BinaryFunc<X86BinaryOpType::kSUB, Float8, 8>;
                break;
            case X86BinaryOpType::kMUL :
                binary_func_ = BinaryFunc<X86BinaryOpType::kMUL, Float8, 8>;
                break;
            case X86BinaryOpType::kDIV :
                binary_func_ = BinaryFunc<X86BinaryOpType::kDIV, Float8, 8>;
                break;
            case X86BinaryOpType::kMAX :
                binary_func_ = BinaryFunc<X86BinaryOpType::kMAX, Float8, 8>;
                break;
            case X86BinaryOpType::kMIN :
                binary_func_ = BinaryFunc<X86BinaryOpType::kMIN, Float8, 8>;
                break;

            default :
                LOGE("Error, unknown binary op_type\n");
                return TNNERR_LAYER_ERR;
        }
    } else if (arch_ == sse42) {
        switch(op_type_) {
            case X86BinaryOpType::kADD :
                binary_func_ = BinaryFunc<X86BinaryOpType::kADD, Float4, 4>;
                break;
            case X86BinaryOpType::kSUB :
                binary_func_ = BinaryFunc<X86BinaryOpType::kSUB, Float4, 4>;
                break;
            case X86BinaryOpType::kMUL :
                binary_func_ = BinaryFunc<X86BinaryOpType::kMUL, Float4, 4>;
                break;
            case X86BinaryOpType::kDIV :
                binary_func_ = BinaryFunc<X86BinaryOpType::kDIV, Float4, 4>;
                break;
            case X86BinaryOpType::kMAX :
                binary_func_ = BinaryFunc<X86BinaryOpType::kMAX, Float4, 4>;
                break;
            case X86BinaryOpType::kMIN :
                binary_func_ = BinaryFunc<X86BinaryOpType::kMIN, Float4, 4>;
                break;

            default :
                LOGE("Error, unknown binary op_type\n");
                return TNNERR_LAYER_ERR;
        }
    }
    switch(op_type_) {
        case X86BinaryOpType::kADD :
            binary_general_func_ = BinaryGeneral<X86BinaryOpType::kADD>;
            break;
        case X86BinaryOpType::kSUB :
            binary_general_func_ = BinaryGeneral<X86BinaryOpType::kSUB>;
            break;
        case X86BinaryOpType::kMUL :
            binary_general_func_ = BinaryGeneral<X86BinaryOpType::kMUL>;
            break;
        case X86BinaryOpType::kDIV :
            binary_general_func_ = BinaryGeneral<X86BinaryOpType::kDIV>;
            break;
        case X86BinaryOpType::kMAX :
            binary_general_func_ = BinaryGeneral<X86BinaryOpType::kMAX>;
            break;
        case X86BinaryOpType::kMIN :
            binary_general_func_ = BinaryGeneral<X86BinaryOpType::kMIN>;
            break;

        default :
            LOGE("Error, unknown binary op_type\n");
            return TNNERR_LAYER_ERR;
    }

    return TNN_OK;
}

// if reshape, reset input_shapes and broadcast type
Status X86BinaryOpLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<MultidirBroadcastLayerParam *>(param_);
    CHECK_PARAM_NULL(layer_param);
    auto layer_res = dynamic_cast<EltwiseLayerResource *>(resource_);

    // prepare input shapes
    input_shapes_.clear();
    input_shapes_.reserve(4);
    auto output = outputs[0];
    auto output_dims = output->GetBlobDesc().dims;

    if (layer_res && inputs.size() == 1) {
        DimsVector input_shape0 = inputs[0]->GetBlobDesc().dims;
        if (layer_param->weight_input_index == 0) {
            // bias as another input
            input_shapes_.push_back(layer_res->element_shape);
            input_shapes_.push_back(input_shape0);
        } else {
            input_shapes_.push_back(input_shape0);
            input_shapes_.push_back(layer_res->element_shape);
        }
    } else {
        if (inputs.size() == 1) {
            input_shapes_.push_back(inputs[0]->GetBlobDesc().dims);
            input_shapes_.push_back(inputs[0]->GetBlobDesc().dims);
        } else {
            for (size_t inid = 0; inid < inputs.size(); inid++) {
                input_shapes_.push_back(inputs[inid]->GetBlobDesc().dims);
            }
        }
    }

    btype_ = BroadcastTypeUnknown;
    // check broadcast type is general or other optimized ncxhwx types
    // if type is general, go to nchw general impl
    DimsVector input_pad_shape;
    input_pad_shape.resize(output_dims.size());
    for (int i = 0; i < input_shapes_.size(); i++) {
        int pad_size = output_dims.size() - input_shapes_[i].size();
        PadShape(pad_size, output_dims.size(), input_pad_shape, input_shapes_[i]);
        BroadCastTypeFilter(output_dims, input_pad_shape, btype_);
        if (btype_ == BroadcastTypeGeneral) {
            break;
        }
    }

    return TNN_OK;
}

Status X86BinaryOpLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<MultidirBroadcastLayerParam *>(param_);
    if (!layer_param) {
        LOGE("Error: layer param is nil\n");
        return Status(TNNERR_PARAM_ERR, "Error: layer param is nil");
    }
    auto layer_res = dynamic_cast<EltwiseLayerResource *>(resource_);

    std::vector<float *> input_ptrs;
    input_ptrs.reserve(4);
    auto output = outputs[0];
    auto dims   = output->GetBlobDesc().dims;

    if (layer_res && inputs.size() == 1) {
        DimsVector input_shape0 = inputs[0]->GetBlobDesc().dims;
        // prepare input ptrs and shapes
        if (layer_param->weight_input_index == 0) {
            // bias as another input
            input_ptrs.push_back(layer_res->element_handle.force_to<float *>());

            input_ptrs.push_back(handle_ptr<float *>(inputs[0]->GetHandle()));
        } else {
            input_ptrs.push_back(handle_ptr<float *>(inputs[0]->GetHandle()));

            input_ptrs.push_back(layer_res->element_handle.force_to<float *>());
        }
    } else {
        if (inputs.size() == 1) {
            input_ptrs.push_back(handle_ptr<float *>(inputs[0]->GetHandle()));
            input_ptrs.push_back(handle_ptr<float *>(inputs[0]->GetHandle()));
        } else {
            for (size_t inid = 0; inid < inputs.size(); inid++) {
                input_ptrs.push_back(handle_ptr<float *>(inputs[inid]->GetHandle()));
            }
        }
    }

    if (btype_ == BroadcastTypeUnknown) {
        LOGE("Error: unknown broadcast type\n");
        return Status(TNNERR_LAYER_ERR, "Error: Binary layer unknown broadcast type");
    } else if (btype_ == BroadcastTypeGeneral) {
        auto output_ptr = handle_ptr<float *>(output->GetHandle());
        binary_general_func_(dims, input_shapes_, output_ptr, input_ptrs);
    } else {
        auto output_ptr = handle_ptr<float *>(output->GetHandle());
        auto input0_ptr = reinterpret_cast<float *>(input_ptrs[0]);
        auto input1_ptr = reinterpret_cast<float *>(input_ptrs[1]);

        // input0_shape != output_shape && input1_shape != output_shape -> general impl
        if (!DimsVectorUtils::Equal(dims, input_shapes_[0]) &&
            !DimsVectorUtils::Equal(dims, input_shapes_[1])) {
            std::vector<DimsVector> shapes_tmp = {input_shapes_[0], input_shapes_[1]};
            std::vector<float *> ptrs_tmp = {input0_ptr, input1_ptr};

            binary_general_func_(dims, shapes_tmp, output_ptr, ptrs_tmp);
        } else {
            DimsVector input0_pad_shape, input1_pad_shape;
            input0_pad_shape.resize(dims.size());
            input1_pad_shape.resize(dims.size());
            PadShape(dims.size() - input_shapes_[0].size(), dims.size(), input0_pad_shape, input_shapes_[0]);
            PadShape(dims.size() - input_shapes_[1].size(), dims.size(), input1_pad_shape, input_shapes_[1]);

            binary_func_(output_ptr, input0_ptr, input1_ptr, input0_pad_shape, input1_pad_shape, output->GetBlobDesc().dims);
        }

        for (int i = 2; i < input_ptrs.size(); i++) {
            DimsVector input0_pad_shape;
            auto input_ptr = reinterpret_cast<float *>(input_ptrs[i]);
            PadShape(dims.size() - input_shapes_[i].size(), dims.size(), input0_pad_shape, input_shapes_[i]);
            binary_func_(output_ptr, output_ptr, input_ptr, dims, input0_pad_shape, output->GetBlobDesc().dims);
        }
    }

    return TNN_OK;
}

}  // namespace TNN_NS
