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

#ifndef TNN_SOURCE_TNN_TRAIN_SOLVER_SOLVER_LAYER_H_
#define TNN_SOURCE_TNN_TRAIN_SOLVER_SOLVER_LAYER_H_

#include "tnn/layer/base_layer.h"
#include "tnn/train/training_info.h"

namespace TNN_NS {

class SolverLayer : public BaseLayer {
public:
    explicit SolverLayer(LayerType ignore);

    virtual ~SolverLayer();

    virtual Status Init(Context *context, LayerParam *param, LayerResource *resource, std::vector<Blob *> &inputs,
                        std::vector<Blob *> &outputs, AbstractDevice *device, bool enable_const_folder = true);

    Status SetTrainableResources(std::vector<RawBuffer *> trainable);

protected:
    virtual Status InferOutputShape(bool ignore_error = false);

private:
    RuntimeTrainingInfo training_info_;
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_TRAIN_SOLVER_SOLVER_LAYER_H_
