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

#include "tnn/network/torch/SegmentedBlock.h"
#include "tnn/interpreter/default_model_interpreter.h"

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
        return res;
    }
    c10::IValue& val = optional_ivalue.value();
    auto optional_res = val.toOptional<T>();
    if (!optional_res) {
        // MNN_ERROR("getValue: value is None.");
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
    const auto shapes = tensor.sizes().vec();
    const auto strides = tensor.strides().vec();
    shape.resize(shapes.size());
    for (int i = 0; i < shapes.size(); i++) {
        shape[i] = static_cast<int>(shapes[i]);
    }
    auto scalarType = tensor.scalar_type();
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

class TorchOpConverter {
public:
    virtual bool IsSupported(const torch::jit::Node *node) {return true;};
    virtual Status Convert(const torch::jit::Node *node, LayerInfo *layer_info, LayerResource **layer_resouce) = 0;
};

std::map<std::string, std::shared_ptr<TorchOpConverter>>& GetGlobalTorchConvertMap();

template <typename T>
class TypeTorchOpConverterRegister {
public:
    explicit TypeTorchOpConverterRegister(std::string type) {
        GetGlobalTorchConvertMap()[type] = shared_ptr<T>(new T);
    }
};

#define REGISTER_TORCH_OP_CONVERTER(type_string, op_type)                                                               \
    TypeTorchOpConverterRegister<type_string##TorchConverter> g_##op_type##_resource_register(#op_type);

}
}
