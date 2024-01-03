#include "test/unit_test/grad_test/test_tool.h"

namespace TNN_NS {

Input::Input(const std::string& name_, const DimsVector& shape_, const std::vector<float>& data_,
             DataType dtype_, bool is_const_)
    : name(name_), shape(shape_), dtype(dtype_), is_const(is_const_) {
    if (dtype == DATA_TYPE_INT32) {
        data       = std::shared_ptr<int32_t>(new int32_t[data_.size()], [](int32_t* p) { delete[] p; });
        int32_t* p = reinterpret_cast<int32_t*>(data.get());
        for (int i = 0; i < data_.size(); i++) {
            p[i] = static_cast<int32_t>(data_[i]);
        }
    } else if (dtype == DATA_TYPE_FLOAT) {
        data = std::shared_ptr<float>(new float[data_.size()], [](float* p) { delete[] p; });
        memcpy(data.get(), data_.data(), data_.size() * sizeof(float));
    }
}

size_t Input::DataBytes() {
    if (dtype == DATA_TYPE_INT32) {
        return sizeof(int32_t) * Size();
    }
    return sizeof(float) * Size();
}

size_t Input::Size() {
    return DimsVectorUtils::Count(shape);
}

std::shared_ptr<AbstractModelInterpreter> CreateInterpreter(
    const std::vector<Input>& net_inputs, const std::set<std::string>& net_outputs, const std::vector<Layer>& layers,
    const std::map<std::string, std::shared_ptr<RawBuffer>>& consts) {
    auto interpreter                             = CreateModelInterpreter(MODEL_TYPE_TNN);
    DefaultModelInterpreter* default_interpreter = dynamic_cast<DefaultModelInterpreter*>(interpreter);
    if (!default_interpreter) {
        LOGE("tnn interpreter is nullptr\n");
        return nullptr;
    }

    // net_structure 只需填充 inputs_shape_map、input_data_type_map、outputs、layers、blobs
    // net_resource 只需要填充resource_map、constant_map
    NetStructure* net_structure = default_interpreter->GetNetStructure();
    NetResource* net_resource   = default_interpreter->GetNetResource();

    net_structure->outputs = net_outputs;

    for (const auto& net_input : net_inputs) {
        net_structure->inputs_shape_map[net_input.name]    = net_input.shape;
        net_structure->input_data_type_map[net_input.name] = net_input.dtype;
    }

    for (const auto& layer : layers) {
        net_structure->layers.push_back(std::make_shared<LayerInfo>(layer.info));
        net_structure->blobs.insert(layer.info.inputs.begin(), layer.info.inputs.end());
        net_structure->blobs.insert(layer.info.outputs.begin(), layer.info.outputs.end());
        if (layer.resource) {
            net_resource->resource_map[layer.info.name] = layer.resource;
        }
    }
    net_resource->constant_map = consts;

    return std::shared_ptr<AbstractModelInterpreter>(interpreter);
}

std::shared_ptr<Instance> CreateInstance(const NetworkConfig& network_config, const std::vector<Input>& net_inputs,
                                         const std::set<std::string>& net_outputs, const std::vector<Layer>& layers,
                                         const std::map<std::string, std::shared_ptr<RawBuffer>>& consts) {
    auto interpreter                             = CreateInterpreter(net_inputs, net_outputs, layers, consts);
    DefaultModelInterpreter* default_interpreter = dynamic_cast<DefaultModelInterpreter*>(interpreter.get());

    auto instance = std::make_shared<Instance>(network_config, ModelConfig());
    Status status = instance->Init(interpreter, default_interpreter->GetNetStructure()->inputs_shape_map);
    if (status != TNN_OK) {
        LOGE("tnn init device instance failed (%s)\n", status.description().c_str());
        return nullptr;
    }
    return instance;
}

std::map<std::string, std::shared_ptr<Mat>> Blob2Mat(std::shared_ptr<Instance> instance, const BlobMap& blobs) {
    std::map<std::string, std::shared_ptr<Mat>> result;

    void* command_queue;
    Status ret = instance->GetCommandQueue(&command_queue);
    if (ret != TNN_OK) {
        LOGE("get device command queue failed (%s)\n", ret.description().c_str());
        return result;
    }

    for (const auto& [name, blob] : blobs) {
        auto mat = std::make_shared<Mat>(DEVICE_NAIVE, NCHW_FLOAT, blob->GetBlobDesc().dims);
        BlobConverter blob_converter(blob);
        ret = blob_converter.ConvertToMat(*mat, MatConvertParam(), command_queue);
        if (ret != TNN_OK) {
            LOGE("ConvertToMat failed (%s)\n", ret.description().c_str());
        } else {
            result[name] = mat;
        }
    }
    return result;
}

std::shared_ptr<Mat> Input2Mat(const Input& input) {
    if (input.dtype == DATA_TYPE_INT32) {
        return std::make_shared<Mat>(DEVICE_ARM, NC_INT32, input.shape, input.data.get());
    } else if (input.dtype == DATA_TYPE_FLOAT) {
        return std::make_shared<Mat>(DEVICE_ARM, NCHW_FLOAT, input.shape, input.data.get());
    } else {
        return nullptr;
    }
}

std::shared_ptr<RawBuffer> Input2Rawbuffer(const Input& input) {
    if (input.dtype == DATA_TYPE_INT32) {
        return std::make_shared<RawBuffer>(sizeof(int32_t) * DimsVectorUtils::Count(input.shape),
                                           (char*)input.data.get(), input.shape, input.dtype);
    } else if (input.dtype == DATA_TYPE_FLOAT) {
        return std::make_shared<RawBuffer>(sizeof(float) * DimsVectorUtils::Count(input.shape), (char*)input.data.get(),
                                           input.shape, input.dtype);
    } else {
        return nullptr;
    }
}

std::string Mat2String(std::shared_ptr<Mat> mat) {
    std::string result;
    result.append("type=").append(std::to_string(mat->GetMatType()));
    result.append("; dim=").append(Join(mat->GetDims()));
    int count = TNN_NS::DimsVectorUtils::Count(mat->GetDims());

    if (mat->GetMatType() == TNN_NS::MatType::NCHW_FLOAT) {
        float* buf = static_cast<float*>(mat->GetData());
        std::vector<float> data(buf, buf + count);
        result.append("; data=").append(Join(data));
    } else if (mat->GetMatType() == TNN_NS::MatType::NC_INT32) {
        int32_t* buf = static_cast<int32_t*>(mat->GetData());
        std::vector<int32_t> data(buf, buf + count);
        result.append("; data=").append(Join(data));
    }
    return result;
}

void PrintMat(const std::map<std::string, std::shared_ptr<Mat>>& mats) {
    std::string str;
    for (const auto& [name, mat] : mats) {
        str.clear();
        str.append("name=").append(name).append("; ").append(Mat2String(mat));
        LOGI("Mat: %s\n", str.c_str());
    }
}

bool Equal(float* v0, float* v1, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (std::abs(v1[i] - v0[i]) > 1e-4) {
            return false;
        }
    }
    return true;
}

}  // namespace TNN_NS
