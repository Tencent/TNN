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

#include "rknpu_base_layer.h"
#include "rknpu_utils.h"
#ifndef TNN_SOURCE_TNN_DEVICE_RK_NPU_CONVERT_RKNPU_BATCH_NORM_LAYER_IMPL_H_
#define TNN_SOURCE_TNN_DEVICE_RK_NPU_CONVERT_RKNPU_BATCH_NORM_LAYER_IMPL_H_

namespace TNN_NS {

class RknpuBatchNormImplLayer : public RknpuBaseLayer {
public:
    RknpuBatchNormImplLayer(LayerType layer_type) : RknpuBaseLayer(layer_type){};
    virtual ~RknpuBatchNormImplLayer() {}

protected:
    std::vector<float> mean_data;
    std::vector<float> variance_data;
    std::vector<float> share_scale_data;
    std::vector<float> share_bias_data;
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_RK_NPU_CONVERT_RKNPU_BATCH_NORM_LAYER_IMPL_H_
