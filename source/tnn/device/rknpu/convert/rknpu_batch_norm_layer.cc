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

#include <tnn/utils/data_type_utils.h>

#include "rknpu_base_layer.h"
#include "rknpu_batch_norm_layer_impl.h"
#include "rknpu_utils.h"

namespace TNN_NS {

class RknpuBatchNormLayer : public RknpuBatchNormImplLayer {
public:
    RknpuBatchNormLayer(LayerType ignore) : RknpuBatchNormImplLayer(LAYER_BATCH_NORM){};
    virtual ~RknpuBatchNormLayer() {}

protected:
    virtual Status Convert() {
        auto resource = dynamic_cast<BatchNormLayerResource *>(resource_);
        if (!resource) {
            return Status(TNNERR_MODEL_ERR, "Error: BatchNorm layer resource is nil");
        }

        auto param = dynamic_cast<BatchNormLayerParam *>(param_);

        // channel is the 1 element of NCHW
        int channel = input_ops_[0]->GetDims()[1];
        bool share_channel =
            resource->scale_handle.GetBytesSize() == DataTypeUtils::GetBytesSize(resource->scale_handle.GetDataType());
        auto *scale_data = resource->scale_handle.force_to<float *>();
        auto *bias_data  = resource->bias_handle.force_to<float *>();

        for (int i = 0; i < channel; i++) {
            mean_data.push_back(0.0f);
            variance_data.push_back(1.0f);
            if (share_channel) {
                share_scale_data.push_back(scale_data[0]);
                share_bias_data.push_back(bias_data[0]);
            }
        }

        Status ret = TNN_OK;
        std::vector<std::shared_ptr<rk::nn::Tensor>> inputs;

        // input
        inputs.push_back(input_ops_[0]);

        std::vector<int> shape = {channel};

        // out = scale * ((in - mean) / variance) + bias

        // mean
        auto mean_const = RknpuUtils::CreateRknnTensor(graph_, layer_name_ + "_mean", shape, mean_data.data(),
                                                       rk::nn::TensorRole::CONST, DATA_TYPE_FLOAT);
        inputs.push_back(mean_const);

        // variance
        auto variance_const = RknpuUtils::CreateRknnTensor(
            graph_, layer_name_ + "_variance", shape, variance_data.data(), rk::nn::TensorRole::CONST, DATA_TYPE_FLOAT);
        inputs.push_back(variance_const);

        // scale & bias
        if (share_channel) {
            auto scale_const =
                RknpuUtils::CreateRknnTensor(graph_, layer_name_ + "_scale", shape, share_scale_data.data(),
                                             rk::nn::TensorRole::CONST, DATA_TYPE_FLOAT);
            auto bias_const = RknpuUtils::CreateRknnTensor(graph_, layer_name_ + "_bias", shape, share_bias_data.data(),
                                                           rk::nn::TensorRole::CONST, DATA_TYPE_FLOAT);
            inputs.push_back(scale_const);
            inputs.push_back(bias_const);
        } else {
            auto scale_const = RknpuUtils::CreateRknnTensor(graph_, layer_name_ + "_scale", shape, scale_data,
                                                            rk::nn::TensorRole::CONST, DATA_TYPE_FLOAT);

            auto bias_const = RknpuUtils::CreateRknnTensor(graph_, layer_name_ + "_bias", shape, bias_data,
                                                           rk::nn::TensorRole::CONST, DATA_TYPE_FLOAT);
            inputs.push_back(scale_const);
            inputs.push_back(bias_const);
        }

        // output
        ADD_OUTPUT_OP();

        rk::nn::BatchNormAttr attrs;
        attrs.eps = (param != NULL) ? param->eps : 0.000001;

        graph_->AddOperator(rk::nn::OperatorType::BATCH_NORM, inputs, output_ops_, &attrs);

        return ret;
    }
};

REGISTER_RKNPU_LAYER(BatchNorm, LAYER_BATCH_NORM)

}  // namespace TNN_NS