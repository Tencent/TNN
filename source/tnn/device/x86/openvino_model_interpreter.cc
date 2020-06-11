// Copyright 2019 Tencent. All Rights Reserved

#include "openvino_model_interpreter.h"

#include <fstream>

#include <inference_engine.hpp>

namespace TNN_NS {

OpenVINOModelInterpreter::OpenVINOModelInterpreter() {
    // load openvino plugin 
}

OpenVINOModelInterpreter::~OpenVINOModelInterpreter() {
    network_ = {};
}

Status OpenVINOModelInterpreter::Interpret(std::vector<std::string> params) {
    std::string xml_content = params[0];
    std::string model_path = params[1];

    std::ifstream read_stream;
    read_stream.open(model_path, std::ios::binary | std::ios::ate);
    if (!read_stream || !read_stream.is_open() || !read_stream.good()) {
        read_stream.close();
        return TNNERR_LOAD_MODEL;
    }

    size_t model_size = read_stream.tellg();
    read_stream.seekg(0, std::ios::beg);

    std::vector<char> buffer(model_size);
    read_stream.read(buffer.data(), model_size);

    try {

        InferenceEngine::TBlob<uint8_t>::Ptr weightsPtr(
            new InferenceEngine::TBlob<uint8_t>(
                InferenceEngine::TensorDesc(
                    InferenceEngine::Precision::U8, {model_size}, InferenceEngine::Layout::C
                )
            )
        );
        weightsPtr->allocate();

        memcpy(weightsPtr->buffer(), buffer.data(), model_size);
        network_ = ie_.ReadNetwork(xml_content, weightsPtr);

    } catch (const InferenceEngine::details::InferenceEngineException& ex) {
        return TNNERR_LOAD_MODEL;
    }

    auto inputInfo = network_.getInputsInfo();
    for(auto inputInfoItem : inputInfo ) {
        inputInfoItem.second->setPrecision(InferenceEngine::Precision::FP32);
        inputInfoItem.second->setLayout(InferenceEngine::Layout::NCHW);
    }

    return TNN_OK;
}

InferenceEngine::CNNNetwork OpenVINOModelInterpreter::GetCNNNetwork() {
    return network_;
}

TypeModelInterpreterRegister<TypeModelInterpreterCreator<OpenVINOModelInterpreter>> g_openvino_model_interpreter_register(MODEL_TYPE_OPENVINO);

}  // namespace TNN_NS
