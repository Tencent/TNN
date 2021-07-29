//
// Created by 李烨 on 20/7/20.
//
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

#include "graph/attr_value.h"
#include "npu_base_layer_convert.h"
#include "npu_reduce_layer_convert.h"
#include "npu_utils.h"

namespace TNN_NS {

class NpuReduceLogSumExpLayer : public NpuReduceLayer {
public:
    NpuReduceLogSumExpLayer(LayerType ignore) : NpuReduceLayer(LAYER_REDUCE_LOG_SUM_EXP) {}
    ~NpuReduceLogSumExpLayer() {}

protected:
    Status Convert() {
        if (!(NpuUtils::VersionCompare(npu_version_, "100.500.xxx.xxx", VCT_BIGEQUAL) &&
              ((NpuUtils::VersionCompare(npu_version_, "100.500.010.011", VCT_BIGEQUAL) &&
                NpuUtils::VersionCompare(npu_version_, "100.500.010.999", VCT_SMALLER)) ||
               (NpuUtils::VersionCompare(npu_version_, "100.500.011.011", VCT_BIGEQUAL) &&
                NpuUtils::VersionCompare(npu_version_, "100.500.011.999", VCT_SMALLER)) ||
               (NpuUtils::VersionCompare(npu_version_, "100.500.012.011", VCT_BIGEQUAL) &&
                NpuUtils::VersionCompare(npu_version_, "100.500.012.999", VCT_SMALLER))))) {
            LOGE("ReduceLogSumExp is supported from 100.500.010.011, but the device version is %s)\n", npu_version_.c_str());
            return Status(TNNERR_LAYER_ERR, "Error: ReduceLogSumExp is not support in this rom version");
        }

        NpuReduceLayer::GetReduceParam();

        std::vector<int64_t> axes(axes_.begin(), axes_.end());

        auto output = std::make_shared<hiai::op::ReduceLogSumExp>(outputs_name_[0]);
        output->set_input_x(*input_ops_[0]->GetOperator());
        output->set_attr_axes(axes);
        output->set_attr_keepdims(keep_dims_);
        ADD_OUTPUT_OP(output)
    }
};

REGISTER_NPU_LAYER(ReduceLogSumExp, LAYER_REDUCE_LOG_SUM_EXP)

}  // namespace TNN_NS
