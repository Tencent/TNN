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

#ifndef TNN_SOURCE_TNN_DEVICE_OPENCL_ACC_OPENCL_REDUCE_LAYER_ACC_H_
#define TNN_SOURCE_TNN_DEVICE_OPENCL_ACC_OPENCL_REDUCE_LAYER_ACC_H_

#include "tnn/device/opencl/acc/opencl_layer_acc.h"

namespace TNN_NS {

class OpenCLReduceLayerAcc : public OpenCLLayerAcc {
public:
    virtual Status Init(Context *context, LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                        const std::vector<Blob *> &outputs) override;

    virtual ~OpenCLReduceLayerAcc() override;

    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;

private:
    virtual std::set<std::string> CreateBuildOptions() = 0;
    bool run_local_work_ = false;
};

#define DECLARE_OPENCL_REDUCE_ACC(type_string)                                                                         \
    class OpenCL##type_string##LayerAcc : public OpenCLReduceLayerAcc {                                                \
    public:                                                                                                            \
        virtual Status Init(Context *context, LayerParam *param, LayerResource *resource,                              \
                            const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;           \
        virtual ~OpenCL##type_string##LayerAcc() override;                                                             \
                                                                                                                       \
    private:                                                                                                           \
        virtual std::set<std::string> CreateBuildOptions() override;                                                   \
    }

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_OPENCL_ACC_OPENCL_REDUCE_LAYER_ACC_H_
