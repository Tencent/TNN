// Copyright 2019 Tencent. All Rights Reserved

#include "atlas_device_utils.h"
#include <stdio.h>
#include "atlas_utils.h"
#include "hiaiengine/ai_memory.h"
#include "hiaiengine/c_graph.h"
#include "tnn/core/macro.h"

namespace TNN_NS {

std::shared_ptr<hiai::IAITensor> CreateTensors(
    const hiai::TensorDimension& tensor_dims,
    std::vector<std::shared_ptr<uint8_t>>& buffer_vec) {
    hiai::AITensorDescription tensor_desc =
        hiai::AINeuralNetworkBuffer::GetDescription();

    uint8_t* buffer      = nullptr;
    HIAI_StatusT get_ret = hiai::HIAIMemory::HIAI_DMalloc(
        tensor_dims.size, (void*&)buffer, hiai::MALLOC_DEFAULT_TIME_OUT,
        hiai::HIAI_MEMORY_ATTR_MANUAL_FREE);
    if (HIAI_OK != get_ret || nullptr == buffer) {
        HIAI_ENGINE_LOG(
            HIAI_IDE_ERROR,
            "[CreateTensors] DMalloc buffer error (size=%d) (ret=0x%x)!\n",
            tensor_dims.size, get_ret);
        return std::shared_ptr<hiai::IAITensor>(nullptr);
    }

    // uint8_t* buffer = (uint8_t*)HIAI_DVPP_DMalloc(tensor_dims.size);
    // if (nullptr == buffer) {
    //    HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "[CreateTensors] HIAI_DVPP_DMalloc
    //    failed."); return std::shared_ptr<hiai::IAITensor>(nullptr);
    //}
    HIAI_ENGINE_LOG(HIAI_IDE_INFO,
                    "[CreateTensors] buffer addr: %lx buffer size: %d\n",
                    (unsigned long)buffer, tensor_dims.size);

    // buffer_vec.push_back(std::shared_ptr<uint8_t>(buffer, HIAI_DVPP_DFree));
    buffer_vec.push_back(
        std::shared_ptr<uint8_t>(buffer, hiai::HIAIMemory::HIAI_DFree));

    std::shared_ptr<hiai::IAITensor> outputTensor =
        hiai::AITensorFactory::GetInstance()->CreateTensor(tensor_desc, buffer,
                                                           tensor_dims.size);
    std::shared_ptr<hiai::AINeuralNetworkBuffer> nn_tensor =
        std::static_pointer_cast<hiai::AINeuralNetworkBuffer>(outputTensor);
    nn_tensor->SetName(tensor_dims.name);
    nn_tensor->SetNumber(tensor_dims.n);
    nn_tensor->SetChannel(tensor_dims.c);
    nn_tensor->SetHeight(tensor_dims.h);
    nn_tensor->SetWidth(tensor_dims.w);
    nn_tensor->SetData_type(tensor_dims.data_type);
    return outputTensor;
}

HIAI_StatusT CreatIOTensors(
    const std::vector<hiai::TensorDimension>& tensor_dims,
    std::vector<std::shared_ptr<hiai::IAITensor>>& tensor_vec,
    std::vector<std::shared_ptr<uint8_t>>& buffer_vec) {
    for (auto& tensorDim : tensor_dims) {
        auto tensor = CreateTensors(tensorDim, buffer_vec);
        if (nullptr == tensor) {
            return HIAI_ERROR;
        }
        tensor_vec.push_back(tensor);
    }
    return HIAI_OK;
}

}  // namespace TNN_NS
