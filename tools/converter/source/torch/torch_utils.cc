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

#include "torch/torch_utils.h"

#include "tnn/interpreter/raw_buffer.h"
#include "tnn/utils/data_type_utils.h"
#include "torch/torch.h"

namespace TNN_CONVERTER {

TNN_NS::DataType TorchDataType2TnnDataType(at::ScalarType scalar_type) {
    switch (scalar_type) {
        case at::ScalarType::Char:
            return TNN_NS::DATA_TYPE_INT8;
        case at::ScalarType::Int:
            return TNN_NS::DATA_TYPE_INT32;
        case at::ScalarType::Long:
            return TNN_NS::DATA_TYPE_INT64;
        case at::ScalarType::Half:
            return TNN_NS::DATA_TYPE_HALF;
        case at::ScalarType::Float:
            return TNN_NS::DATA_TYPE_FLOAT;
        case at::ScalarType::Double:
            return TNN_NS::DATA_TYPE_FLOAT;
        default: {
            LOGI("TorchDataType2TnnDataType does not know exactly data type, so use default(float)\n");
            return TNN_NS::DATA_TYPE_FLOAT;
        }
    }
}

std::string GetRealOpType(const torch::jit::Node *node) {
    const auto kind = node->kind();
    // custom op
    if (!(kind.is_attr() || kind.is_aten() || kind.is_cuda() || kind.is_prim() || kind.is_onnx() || kind.is_user() ||
          kind.is_caffe2() || kind.is_dimname())) {
        return "__custom__";
    }
    std::string opType(kind.toUnqualString());
    // convert _xxx_ to xxx
    int last  = opType.size() - 1;
    int last2 = last - 1;
    if (last > 0 && opType[last] == '_' && opType[last2] != '_') {
        opType = opType.substr(0, opType.size() - 1);
    }
    if (opType.size() > 2 && opType[0] == '_' && opType[1] != '_') {
        opType = opType.substr(1, opType.size() - 1);
    }
    // distinguish overload function
    auto symb = c10::Symbol::fromQualString("attr::mnn_tag");
    if (node->hasAttribute(symb)) {
        opType += ("_" + node->s(symb));
    }
    return opType;
}

bool DealPrime(const torch::jit::Node *node) {
    std::string opType = GetRealOpType(node);
    switch (node->kind()) {
        case at::prim::Constant:
        case at::prim::ListUnpack:
        case at::prim::TupleConstruct:
        case at::prim::Uninitialized:
            return true;
        default:
            break;
    }
    if (opType == "If") {
        if (!node->outputs().empty()) {
            return false;
        }
        return true;
    }
    if (opType == "Loop") {
        return false;
    }
    return false;
}

TNN_NS::RawBuffer CreateRawBufferFromTensor(const at::Tensor &tensor) {
    auto torch_type = tensor.scalar_type();
    auto size       = tensor.numel();
    TNN_NS::DimsVector shape;
    if (torch_type == at::ScalarType::QInt8) {
        auto vec        = GetValue<c10::qint8>(tensor, shape);
        auto bytes_size = size * TNN_NS::DataTypeUtils::GetBytesSize(TNN_NS::DATA_TYPE_INT8);
        auto buffer     = TNN_NS::RawBuffer(bytes_size, reinterpret_cast<char *>(vec.data()), shape);
        buffer.SetDataType(TNN_NS::DATA_TYPE_INT8);
        return buffer;
    } else if (torch_type == at::ScalarType::Float) {
        auto vec        = GetValue<float>(tensor, shape);
        auto bytes_size = size * TNN_NS::DataTypeUtils::GetBytesSize(TNN_NS::DATA_TYPE_FLOAT);
        auto buffer     = TNN_NS::RawBuffer(bytes_size, reinterpret_cast<char *>(vec.data()), shape);
        buffer.SetDataType(TNN_NS::DATA_TYPE_FLOAT);
        return buffer;
    } else if (torch_type == at::ScalarType::Double) {
        auto vec        = GetValue<double>(tensor, shape);
        auto bytes_size = size * TNN_NS::DataTypeUtils::GetBytesSize(TNN_NS::DATA_TYPE_FLOAT);
        auto float_vec  = std::vector<float>(vec.begin(), vec.end());
        auto buffer     = TNN_NS::RawBuffer(bytes_size, reinterpret_cast<char *>(float_vec.data()), shape);
        buffer.SetDataType(tnn::DATA_TYPE_FLOAT);
        return buffer;
    } else if (torch_type == at::ScalarType::Long) {
        auto vec        = GetValue<int64_t>(tensor, shape);
        auto bytes_size = size * TNN_NS::DataTypeUtils::GetBytesSize(TNN_NS::DATA_TYPE_INT32);
        auto res_vec    = std::vector<int>(vec.begin(), vec.end());
        auto buffer     = TNN_NS::RawBuffer(bytes_size, reinterpret_cast<char *>(res_vec.data()), shape);
        buffer.SetDataType(TNN_NS::DATA_TYPE_INT32);
        return buffer;
    } else {
        LOGE("CreateRawBufferFromTensor does not support torch type: %hhd\n", torch_type);
        return TNN_NS::RawBuffer();
    }
}

TNN_NS::RawBuffer CreateRawBufferFromValue(const torch::jit::Value *value) {
    const auto &value_kind = value->type()->kind();
    if (value_kind != c10::TypeKind::TensorType) {
        if (value_kind == c10::TypeKind::FloatType) {
            const auto data = GetValue<float>(value);
            auto buffer     = TNN_NS::RawBuffer(sizeof(float), (char *)(&data), {1});
            buffer.SetDataType(TNN_NS::DATA_TYPE_FLOAT);
            return buffer;
        } else if (value_kind == c10::TypeKind::IntType) {
            int64_t data_int64    = GetValue<int64_t>(value);
            data_int64            = std::max(data_int64, static_cast<int64_t>(INT_MIN));
            data_int64            = std::min(data_int64, static_cast<int64_t>(INT_MAX));
            int data              = static_cast<int>(data_int64);
            TNN_NS::RawBuffer buf = TNN_NS::RawBuffer(4, (char *)(&data), {1});
            buf.SetDataType(TNN_NS::DATA_TYPE_INT32);
            return buf;
        } else if (value_kind == c10::TypeKind::NoneType) {
            LOGE("CreateRawBufferFromValue get value kind is c10::TypeKind::NoneType\n");
            return TNN_NS::RawBuffer();
        } else {
            LOGE("CreateRawBufferFromValue:wrong scalar type\n");
            return TNN_NS::RawBuffer();
        }
    } else {
        const auto tensor = GetValue<at::Tensor>(value).to(at::kCPU);
        int size          = tensor.numel();
        if (!size) {
            LOGE("CreateRawBufferFromValue get 0 size, so create empty RawBuffer\n");
            return TNN_NS::RawBuffer();
        }
        auto torch_type = tensor.scalar_type();
        TNN_NS::DimsVector dims;
        if (torch_type == at::ScalarType::Half) {
            auto vec        = GetValue<at::Half>(tensor, dims);
            auto bytes_size = size * TNN_NS::DataTypeUtils::GetBytesSize(TNN_NS::DATA_TYPE_HALF);
            auto buffer     = TNN_NS::RawBuffer(bytes_size, reinterpret_cast<char *>(vec.data()), dims);
            buffer.SetDataType(TNN_NS::DATA_TYPE_HALF);
            return buffer;
        } else if (torch_type == at::ScalarType::Float) {
            auto vec        = GetValue<float>(value, dims);
            auto bytes_size = size * TNN_NS::DataTypeUtils::GetBytesSize(TNN_NS::DATA_TYPE_FLOAT);
            auto buffer     = TNN_NS::RawBuffer(bytes_size, reinterpret_cast<char *>(vec.data()), dims);
            buffer.SetDataType(TNN_NS::DATA_TYPE_FLOAT);
            return buffer;
        } else if (torch_type == at::ScalarType::Double) {
            auto vec        = GetValue<double>(value, dims);
            auto bytes_size = size * TNN_NS::DataTypeUtils::GetBytesSize(TNN_NS::DATA_TYPE_FLOAT);
            auto float_vec  = std::vector<float>(vec.begin(), vec.end());
            auto buffer     = TNN_NS::RawBuffer(bytes_size, reinterpret_cast<char *>(float_vec.data()), dims);
            buffer.SetDataType(TNN_NS::DATA_TYPE_FLOAT);
            return buffer;
        } else if (torch_type == at::ScalarType::Long) {
            auto vec        = GetValue<int64_t>(value, dims);
            auto bytes_size = size * TNN_NS::DataTypeUtils::GetBytesSize(TNN_NS::DATA_TYPE_INT32);
            auto res_vec    = std::vector<int>(vec.begin(), vec.end());
            auto buffer     = TNN_NS::RawBuffer(bytes_size, reinterpret_cast<char *>(res_vec.data()), dims);
            buffer.SetDataType(TNN_NS::DATA_TYPE_INT32);
            return buffer;
        } else {
            LOGE("getValue:wrong scalar type, so create emtpy RawBuffer\n");
            return TNN_NS::RawBuffer();
        }
    }
}

TNN_NS::DimsVector GetDimsFromValue(const torch::jit::Value *value) {
    const auto tensor = GetValue<at::Tensor>(value);
    auto shape        = tensor.sizes().vec();
    TNN_NS::DimsVector dims(shape.begin(), shape.end());
    return dims;
}

TNN_NS::DimsVector GetDimsFromTensor(const at::Tensor &tensor) {
    const auto shape = tensor.sizes().vec();
    TNN_NS::DimsVector dims(shape.begin(), shape.end());
    return dims;
}

// Get Real effective Input Values of node.
// For Example,
// %68 : int = aten::size(%input_ids.1, %8)
// %seq_length.1 : Tensor = prim::NumToTensor(%68)
// %70 : Tensor = aten::add(%seq_length.1, %36, %8)
// In this paragraph, effective input values of node aten::add
// should be %68, not %seq_length.1.
torch::jit::Value *GetEffectiveInputValue(const torch::jit::Node *node, int idx) {
    torch::jit::Value *ret;
    auto input      = node->input(idx);
    auto input_kind = input->node()->kind();
    if (input_kind == at::prim::NumToTensor || input_kind == at::aten::Int || input_kind == at::aten::contiguous) {
        input = GetEffectiveInputValue(input->node(), 0);
    }
    return input;
}

std::vector<torch::jit::Value *> GetEffectiveInputValues(const torch::jit::Node *node) {
    std::vector<torch::jit::Value *> ret;
    for (int i = 0; i < node->inputs().size(); i++) {
        ret.push_back(GetEffectiveInputValue(node, i));
    }
    return ret;
}

TNN_NS::RawBuffer ConvertRawBuffFromIntToInt8(TNN_NS::RawBuffer &src_buffer) {
    const auto src_buffer_data_type = src_buffer.GetDataType();
    if (src_buffer_data_type == TNN_NS::DATA_TYPE_INT8) {
        return TNN_NS::RawBuffer(src_buffer);
    }
    if (src_buffer_data_type != TNN_NS::DATA_TYPE_INT32) {
        LOGE("Only support raw buffer type is int32_t, but get type %d\n", src_buffer_data_type);
        ASSERT(0);
        return TNN_NS::RawBuffer();
    }
    const auto *src_buffer_ptr   = src_buffer.force_to<int32_t *>();
    const auto &dims             = src_buffer.GetBufferDims();
    const int count              = src_buffer.GetDataCount();
    const int buffer_byte_size   = count * sizeof(int8_t);
    TNN_NS::RawBuffer dst_buffer = TNN_NS::RawBuffer(buffer_byte_size, dims);
    dst_buffer.SetDataType(TNN_NS::DATA_TYPE_INT8);
    auto *dst_buffer_ptr = dst_buffer.force_to<int8_t *>();
    for (int i = 0; i < count; ++i) {
        dst_buffer_ptr[i] = (int8_t)src_buffer_ptr[i];
    }
    return dst_buffer;
}

}  // namespace TNN_CONVERTER
