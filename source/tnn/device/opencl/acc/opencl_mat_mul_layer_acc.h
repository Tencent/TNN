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

#include "tnn/device/opencl/acc/opencl_layer_acc.h"
#include "tnn/device/opencl/acc/opencl_reshape_layer_acc.h"

namespace TNN_NS {

class OpenCLMatMulLayerAcc : public OpenCLLayerAcc {
public:
    virtual Status Init(Context *context, LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                        const std::vector<Blob *> &outputs) override;

    virtual ~OpenCLMatMulLayerAcc() override;

    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;

    virtual Status Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;

private:
    Status InitReshapeLayer(Blob *blob, std::shared_ptr<OpenCLReshapeLayerAcc>& layer,
                            bool &need_reshape, std::vector<Blob *> &reshape_layer_inputs,
                            std::vector<Blob *> &reshape_layer_outputs, std::shared_ptr<Blob>& reshape_blob,
                            int position);
    Status ConvertWeights(float *weights_data_ptr, int weight_w, int weight_h);

private:
    DimsVector matrix_a_dims_ = {};
    DimsVector matrix_b_dims_ = {};
    DimsVector matrix_c_dims_ = {};
    int weight_position_ = 0;
    // input0, input1, output
    std::vector<bool> need_reshape_ = {false, false, false};
    std::vector<std::shared_ptr<OpenCLReshapeLayerAcc> > reshape_layer_acc_ = {nullptr, nullptr, nullptr};
    std::vector<std::vector<Blob *> > reshape_inputs_ = {};
    std::vector<std::vector<Blob *> > reshape_outputs_ = {};
    std::vector<std::shared_ptr<Blob> > reshape_blob_ = {nullptr, nullptr, nullptr};
};

}  // namespace TNN_NS
