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
#include "tnn/utils/dims_vector_utils.h"
#include "tnn/device/x86/acc/Float4.h"
#include "tnn/device/x86/acc/Float8.h"

namespace TNN_NS {

template<X86BinaryOpType type>
float binary_op(const float a, const float b) {
    return a;
}
template<> float binary_op<X86BinaryOpType::kADD>(const float a, const float b) {
    return a + b;
}
template<> float binary_op<X86BinaryOpType::kSUB>(const float a, const float b) {
    return a - b;
}
template<> float binary_op<X86BinaryOpType::kMUL>(const float a, const float b) {
    return a * b;
}
template<> float binary_op<X86BinaryOpType::kDIV>(const float a, const float b) {
    return a / b;
}
template<> float binary_op<X86BinaryOpType::kMAX>(const float a, const float b) {
    return a > b ? a : b;
}
template<> float binary_op<X86BinaryOpType::kMIN>(const float a, const float b) {
    return a < b ? a : b;
}

template<X86BinaryOpType type, typename VEC>
VEC binary_op(const VEC a, const VEC b) {
    return a;
}
template<> Float4 binary_op<X86BinaryOpType::kADD, Float4>(const Float4 a, const Float4 b) {
    return Float4::add(a, b);
}
template<> Float4 binary_op<X86BinaryOpType::kSUB, Float4>(const Float4 a, const Float4 b) {
    return Float4::sub(a, b);
}
template<> Float4 binary_op<X86BinaryOpType::kMUL, Float4>(const Float4 a, const Float4 b) {
    return Float4::mul(a, b);
}
template<> Float4 binary_op<X86BinaryOpType::kDIV, Float4>(const Float4 a, const Float4 b) {
    return Float4::div(a, b);
}
template<> Float4 binary_op<X86BinaryOpType::kMAX, Float4>(const Float4 a, const Float4 b) {
    return Float4::max(a, b);
}
template<> Float4 binary_op<X86BinaryOpType::kMIN, Float4>(const Float4 a, const Float4 b) {
    return Float4::min(a, b);
}
template<> Float8 binary_op<X86BinaryOpType::kADD, Float8>(const Float8 a, const Float8 b) {
    return Float8::add(a, b);
}
template<> Float8 binary_op<X86BinaryOpType::kSUB, Float8>(const Float8 a, const Float8 b) {
    return Float8::sub(a, b);
}
template<> Float8 binary_op<X86BinaryOpType::kMUL, Float8>(const Float8 a, const Float8 b) {
    return Float8::mul(a, b);
}
template<> Float8 binary_op<X86BinaryOpType::kDIV, Float8>(const Float8 a, const Float8 b) {
    return Float8::div(a, b);
}
template<> Float8 binary_op<X86BinaryOpType::kMAX, Float8>(const Float8 a, const Float8 b) {
    return Float8::max(a, b);
}
template<> Float8 binary_op<X86BinaryOpType::kMIN, Float8>(const Float8 a, const Float8 b) {
    return Float8::min(a, b);
}

static void BroadCastInit(const DimsVector &dims, const DimsVector &dims0, const DimsVector &dims1, BroadcastType &type,
                          DimsVector &dims_broadcast, bool &swap_flag) {
    if (DimsVectorUtils::Equal(dims0, dims1)) {
        type = BroadcastTypeNormal;
        dims_broadcast.clear();
    } else if (DimsVectorUtils::Equal(dims0, dims1, 1)) {
        type = BroadcastTypeElement;
        dims_broadcast.clear();
        if (dims0[0] < dims1[0])
            swap_flag = true;
    } else if (DimsVectorUtils::Equal(dims0, dims1, 2)) {
        type = BroadcastTypeHeightWidth;
        dims_broadcast.clear();
        if (dims0[1] < dims1[1])
            swap_flag = true;
    } else if (DimsVectorUtils::Equal(dims0, dims1, 3)) {
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

/*
Binary func with different opreator,
set dims0 full shape, dims1 broadcast shape, so we need to swap input ptrs
*/
template <X86BinaryOpType op_type, typename VEC, int pack>
Status BinaryFunc(float *output_ptr, const float *input0_ptr, const float *input1_ptr, DimsVector &dims0, DimsVector &dims1) {
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

    if (dims_broadcast.size()) {
        type = (dims_broadcast[1] == 1) ? BroadcastTypeSingle : BroadcastTypeChannel;
    }

    size_t count = dims[0] * dims[1] * dims[2] * dims[3];
    size_t batch_stride = dims[1] * dims[2] * dims[3];
    size_t channel_stride = dims[2] * dims[3];

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
                            VEC::saveu(_output_h + h, binary_op<op_type, VEC>(v2, v1));
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
                            VEC::saveu(_output_h + h, binary_op<op_type, VEC>(v1, v2));
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

Status X86BinaryOpLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                              const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    RETURN_ON_NEQ(X86LayerAcc::Init(context, param, resource, inputs, outputs), TNN_OK);
    return TNN_OK;
}

Status X86BinaryOpLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

X86BinaryOpLayerAcc::~X86BinaryOpLayerAcc() {}

Status X86BinaryOpLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<MultidirBroadcastLayerParam *>(param_);
    if (!layer_param) {
        LOGE("Error: layer param is nil\n");
        return Status(TNNERR_PARAM_ERR, "Error: layer param is nil");
    }
    auto layer_res = dynamic_cast<EltwiseLayerResource *>(resource_);

    std::vector<void *> input_ptrs;
    std::vector<DimsVector> input_shapes;
    input_ptrs.reserve(4);
    input_shapes.reserve(4);
    auto output = outputs[0];
    auto dims   = output->GetBlobDesc().dims;

    if (layer_res && inputs.size() == 1) {
        DimsVector input_shape0 = inputs[0]->GetBlobDesc().dims;
        // prepare input ptrs and shapes
        if (layer_param->weight_input_index == 0) {
            // bias as another input
            input_ptrs.push_back(layer_res->element_handle.force_to<void *>());
            input_shapes.push_back(layer_res->element_shape);

            input_ptrs.push_back(inputs[0]->GetHandle().base);
            input_shapes.push_back(input_shape0);
        } else {
            input_ptrs.push_back(inputs[0]->GetHandle().base);
            input_shapes.push_back(input_shape0);

            input_ptrs.push_back(layer_res->element_handle.force_to<void *>());
            input_shapes.push_back(layer_res->element_shape);
        }
    } else {
        if (inputs.size() == 1) {
            input_ptrs.push_back(inputs[0]->GetHandle().base);
            input_ptrs.push_back(inputs[0]->GetHandle().base);
            input_shapes.push_back(inputs[0]->GetBlobDesc().dims);
            input_shapes.push_back(inputs[0]->GetBlobDesc().dims);
        } else {
            for (size_t inid = 0; inid < inputs.size(); inid++) {
                input_ptrs.push_back(inputs[inid]->GetHandle().base);
                input_shapes.push_back(inputs[inid]->GetBlobDesc().dims);
            }
        }
    }

    auto output_ptr = reinterpret_cast<float *>(output->GetHandle().base);
    auto input0_ptr = reinterpret_cast<float *>(input_ptrs[0]);
    auto input1_ptr = reinterpret_cast<float *>(input_ptrs[1]);

    auto binary_func = BinaryFunc<X86BinaryOpType::kADD, Float4, 4>;

    if (arch_ == avx2) {
        switch(op_type_) {
            case X86BinaryOpType::kADD :
                binary_func = BinaryFunc<X86BinaryOpType::kADD, Float8, 8>;
                break;
            case X86BinaryOpType::kSUB :
                binary_func = BinaryFunc<X86BinaryOpType::kSUB, Float8, 8>;
                break;
            case X86BinaryOpType::kMUL :
                binary_func = BinaryFunc<X86BinaryOpType::kMUL, Float8, 8>;
                break;
            case X86BinaryOpType::kDIV :
                binary_func = BinaryFunc<X86BinaryOpType::kDIV, Float8, 8>;
                break;
            case X86BinaryOpType::kMAX :
                binary_func = BinaryFunc<X86BinaryOpType::kMAX, Float8, 8>;
                break;
            case X86BinaryOpType::kMIN :
                binary_func = BinaryFunc<X86BinaryOpType::kMIN, Float8, 8>;
                break;

            default :
                LOGE("Error, unknown binary op_type\n");
                return TNNERR_LAYER_ERR;
        }
    } else if (arch_ == sse42) {
        switch(op_type_) {
            case X86BinaryOpType::kADD :
                binary_func = BinaryFunc<X86BinaryOpType::kADD, Float4, 4>;
                break;
            case X86BinaryOpType::kSUB :
                binary_func = BinaryFunc<X86BinaryOpType::kSUB, Float4, 4>;
                break;
            case X86BinaryOpType::kMUL :
                binary_func = BinaryFunc<X86BinaryOpType::kMUL, Float4, 4>;
                break;
            case X86BinaryOpType::kDIV :
                binary_func = BinaryFunc<X86BinaryOpType::kDIV, Float4, 4>;
                break;
            case X86BinaryOpType::kMAX :
                binary_func = BinaryFunc<X86BinaryOpType::kMAX, Float4, 4>;
                break;
            case X86BinaryOpType::kMIN :
                binary_func = BinaryFunc<X86BinaryOpType::kMIN, Float4, 4>;
                break;

            default :
                LOGE("Error, unknown binary op_type\n");
                return TNNERR_LAYER_ERR;
        }
    }

    binary_func(output_ptr, input0_ptr, input1_ptr, input_shapes[0], input_shapes[1]);

    for (int i = 2; i < input_ptrs.size(); i++) {
        auto input_ptr = reinterpret_cast<float *>(input_ptrs[i]);
        binary_func(output_ptr, output_ptr, input_ptr, dims, input_shapes[i]);
    }

    return TNN_OK;
}

}  // namespace TNN_NS
