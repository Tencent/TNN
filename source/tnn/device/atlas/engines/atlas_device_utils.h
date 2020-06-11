// Copyright 2019 Tencent. All Rights Reserved

#ifndef TNN_SOURCE_DEVICE_ATLAS_ENGINES_ATLAS_DEVICE_UTILS_H_
#define TNN_SOURCE_DEVICE_ATLAS_ENGINES_ATLAS_DEVICE_UTILS_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "hiaiengine/api.h"

namespace TNN_NS {

std::shared_ptr<hiai::IAITensor> CreateTensors(
    const hiai::TensorDimension& tensor_dims,
    std::vector<std::shared_ptr<uint8_t>>& buffer_vec);

HIAI_StatusT CreatIOTensors(
    const std::vector<hiai::TensorDimension>& tensor_dims,
    std::vector<std::shared_ptr<hiai::IAITensor>>& tensor_vec,
    std::vector<std::shared_ptr<uint8_t>>& buffer_vec);

}  // namespace TNN_NS

#endif  // TNN_SOURCE_DEVICE_ATLAS_ENGINES_ATLAS_DEVICE_UTILS_H_
