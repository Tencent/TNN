#pragma once
#include <vector>

#include "tnn/interpreter/abstract_model_interpreter.h"
#include "tnn/core/abstract_network.h"
#include "tnn/core/default_network.h"
#include "tnn/core/blob.h"
#include "tnn/core/blob_manager.h"
#include "tnn/core/common.h"
#include "tnn/core/context.h"
#include "tnn/core/macro.h"
#include "tnn/interpreter/net_resource.h"
#include "tnn/interpreter/net_structure.h"
#include "tnn/layer/base_layer.h"

#include "tnn/network/torch/segment.h"
#include "tnn/network/torch/torch_utils.h"
#include "tnn/interpreter/default_model_interpreter.h"
#include "tnn/utils/data_type_utils.h"

#include <torch/script.h>
#include "c10/util/intrusive_ptr.h"
#include "torch/custom_class.h"

namespace TNN_NS {
namespace conversion {

template <typename T>
static inline T getValue(const torch::jit::Value* value) {
    auto optional_ivalue = toIValue(value);
    T res;
    if (!optional_ivalue) {
        LOGE("getValue:Value cannot be interpret as IValue\n");
        return res;
    }
    c10::IValue& val = optional_ivalue.value();
    auto optional_res = val.toOptional<T>();
    if (!optional_res) {
        LOGE("getValue:IValue is none");
        return res;
    }
    return optional_res.value();
}

template <typename T>
static std::vector<T> getValue(const torch::jit::Value* value, std::vector<int>& shape) {
    std::vector<T> data;
    const auto tensor = getValue<at::Tensor>(value).to(at::kCPU);
    int size = tensor.numel();
    if (!size) {
        return data;
    }
    if (size == 1) {
        data.resize(1);
        shape.resize(1);
        data[0] = tensor.data_ptr<T>()[0];
        shape[0] = 1;
        return data;
    }
    const auto shapes = tensor.sizes().vec();
    const auto strides = tensor.strides().vec();
    shape.resize(shapes.size());
    for (int i = 0; i < shapes.size(); i++) {
        shape[i] = static_cast<int>(shapes[i]);
    }
    data.resize(size);
    int idx = 0;
    std::function<void(int, int)> copyData = [&](int dim, int offset) {
        if (dim == shapes.size()-1) {
            for (int i = 0; i < shapes[dim]; i++) {
                data[idx++] = tensor.data_ptr<T>()[offset + i * strides[dim]];

            }
        } else {
            for (int i = 0; i < shapes[dim]; i++) {
                copyData(dim + 1, offset + i * strides[dim]);
            }
        }
    };
    copyData(0, 0);
    return data;
}

template <typename T>
static std::vector<T> getValue(const at::Tensor &tensor, std::vector<int>& shape) {
    std::vector<T> data;
    int size = tensor.numel();
    if (!size) {
        return data;
    }
    const auto shapes = tensor.sizes().vec();
    const auto strides = tensor.strides().vec();
    shape.resize(shapes.size());
    for (int i = 0; i < shapes.size(); i++) {
        shape[i] = static_cast<int>(shapes[i]);
    }
    data.resize(size);
    int idx = 0;
    std::function<void(int, int)> copyData = [&](int dim, int offset) {
        if (dim == shapes.size()-1) {
            for (int i = 0; i < shapes[dim]; i++) {
                data[idx++] = tensor.data_ptr<T>()[offset + i * strides[dim]];

            }
        } else {
            for (int i = 0; i < shapes[dim]; i++) {
                copyData(dim + 1, offset + i * strides[dim]);
            }
        }
    };
    copyData(0, 0);
    return data;
}

static RawBuffer getValue(const torch::jit::Value* value) {
    const auto& value_kind = value->type()->kind();
    if (value_kind != c10::TypeKind::TensorType) {
        if (value_kind == c10::TypeKind::FloatType) {
            const auto data = getValue<float>(value);
            return RawBuffer(4, (char*)(&data), {1});
        } else if (value_kind == c10::TypeKind::IntType) {
            int64_t data_int64 = getValue<int64_t>(value);
            data_int64 = std::max(data_int64, static_cast<int64_t>(INT_MIN));
            data_int64 = std::min(data_int64, static_cast<int64_t>(INT_MAX));
            int data = static_cast<int>(data_int64);
            RawBuffer buf = RawBuffer(4, (char*)(&data), {1});
            buf.SetDataType(DATA_TYPE_INT32);
            return buf;
        } else if (value_kind == c10::TypeKind::NoneType) {
            return RawBuffer();
        } else {
            LOGE("getValue:wrong scalar type\n");
            return RawBuffer();
        }
    } else {
        const auto tensor = getValue<at::Tensor>(value).to(at::kCPU);
        int size          = tensor.numel();
        if (!size) {
            return RawBuffer();
        }
        DataType data_type;
        auto torch_type = tensor.scalar_type();
        DimsVector dims;
        if (torch_type == at::ScalarType::Half) {
            auto vec        = getValue<at::Half>(tensor, dims);
            auto bytes_size = size * DataTypeUtils::GetBytesSize(DATA_TYPE_HALF);
            auto buffer     = RawBuffer(bytes_size, reinterpret_cast<char*>(vec.data()), dims);
            buffer.SetDataType(DATA_TYPE_HALF);
            return buffer;
        } else if (torch_type == at::ScalarType::Float) {
            auto vec        = getValue<float>(value, dims);
            auto bytes_size = size * DataTypeUtils::GetBytesSize(DATA_TYPE_FLOAT);
            return RawBuffer(bytes_size, reinterpret_cast<char*>(vec.data()), dims);
        } else if (torch_type == at::ScalarType::Double) {
            auto vec        = getValue<double>(value, dims);
            auto bytes_size = size * DataTypeUtils::GetBytesSize(DATA_TYPE_FLOAT);
            auto float_vec  = std::vector<float>(vec.begin(), vec.end());
            return RawBuffer(bytes_size, reinterpret_cast<char*>(float_vec.data()), dims);
        } else if (torch_type == at::ScalarType::Long) {
            auto vec        = getValue<int64_t>(value, dims);
            auto bytes_size = size * DataTypeUtils::GetBytesSize(DATA_TYPE_FLOAT);
            auto float_vec  = std::vector<float>(vec.begin(), vec.end());
            return RawBuffer(bytes_size, reinterpret_cast<char*>(float_vec.data()), dims);
        } else {
            LOGE("getValue:wrong scalar type\n");
        }
    }

    return RawBuffer();
}

class TorchOpConverter {
public:
    virtual bool IsSupported(const torch::jit::Node *node) {return true;};
    virtual Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) = 0;
};

std::map<std::string, std::shared_ptr<TorchOpConverter>>& GetGlobalTorchConvertMap();

template <typename T>
class TypeTorchOpConverterRegister {
public:
    explicit TypeTorchOpConverterRegister(std::string type) {
        GetGlobalTorchConvertMap()[type] = shared_ptr<T>(new T);
    }
};

#define REGISTER_TORCH_OP_CONVERTER(type_string, ns, op_type)                                                               \
    TypeTorchOpConverterRegister<type_string##TorchConverter> g_##ns##op_type##_resource_register(#ns"::"#op_type);

}
}
