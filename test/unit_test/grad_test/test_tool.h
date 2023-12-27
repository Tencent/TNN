#pragma once

#include <sstream>

#include "tnn/core/tnn.h"
#include "tnn/layer/base_layer.h"
#include "tnn/interpreter/default_model_interpreter.h"

namespace TNN_NS {

struct Input {
    std::string name;
    DimsVector shape;
    std::shared_ptr<void> data = nullptr;
    DataType dtype             = DATA_TYPE_FLOAT;
    bool is_const              = false;

    Input(const std::string& name_, const DimsVector& shape_, const std::vector<float>& data_,
          DataType dtype_ = DATA_TYPE_FLOAT, bool is_const_ = false);
    size_t DataBytes();
    size_t Size();
};

struct Layer {
    LayerInfo info;
    std::shared_ptr<LayerResource> resource;
};

// 方便快捷地创建一个TnnInstance
std::shared_ptr<Instance> CreateInstance(const NetworkConfig& network_config, const std::vector<Input>& net_inputs,
                                         const std::set<std::string>& net_outputs, const std::vector<Layer>& layers,
                                         const std::map<std::string, std::shared_ptr<RawBuffer>>& consts);

std::shared_ptr<AbstractModelInterpreter> CreateInterpreter(
    const std::vector<Input>& net_inputs, const std::set<std::string>& net_outputs, const std::vector<Layer>& layers,
    const std::map<std::string, std::shared_ptr<RawBuffer>>& consts);

// 一些测试辅助方法
std::map<std::string, std::shared_ptr<Mat>> Blob2Mat(std::shared_ptr<Instance> instance, const BlobMap& blobs);
std::shared_ptr<Mat> Input2Mat(const Input& input);
std::shared_ptr<RawBuffer> Input2Rawbuffer(const Input& input);
std::string Mat2String(std::shared_ptr<Mat> mat);
void PrintMat(const std::map<std::string, std::shared_ptr<Mat>>& mats);
bool Equal(float* v0, float* v1, size_t n);

template <typename T>
std::string Join(const std::initializer_list<T>& sources, const std::string& delimiters = ",") {
    std::stringstream ss;
    std::string pre = "";
    for (const auto& source : sources) {
        ss << pre << source;
        pre = delimiters;
    }
    return ss.str();
}

template <typename ITERABLE>
std::string Join(const ITERABLE& iterable, const std::string& delimiters = ",") {
    std::stringstream ss;
    std::string pre = "";
    for (auto& scope : iterable) {
        ss << pre << scope;
        pre = delimiters;
    }
    return ss.str();
}

}  // namespace TNN_NS
