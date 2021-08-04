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
#include "tnn/device/opencl/acc/opencl_reshape_layer_acc.h"

namespace TNN_NS {

class OpenCLReduceLayerAcc : public OpenCLLayerAcc {
public:
    virtual Status Init(Context *context, LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                        const std::vector<Blob *> &outputs) override;

    virtual ~OpenCLReduceLayerAcc() override;

    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;

    virtual Status Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;

private:
    virtual std::set<std::string> CreateBuildOptions() = 0;
    Status InitReshapeLayer(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs,
                            DimsVector &reshape_shape, shared_ptr<OpenCLReshapeLayerAcc> &reshape_layer_acc);
    Status CreaterBlob(BlobDesc desc, DimsVector dims, std::shared_ptr<Blob> &blob);
    DimsVector GenerateInputShape(DimsVector &input_dims, int axis);
    DimsVector AfterReduceDims(DimsVector dims, std::vector<int> axis);
    bool run_local_work_      = false;
    bool input_need_reshape_  = false;
    bool output_need_reshape_ = false;

    shared_ptr<OpenCLReshapeLayerAcc> reshape_input_layer_acc_         = nullptr;
    shared_ptr<OpenCLReshapeLayerAcc> reshape_output_layer_acc_        = nullptr;
    std::vector<std::shared_ptr<ReshapeLayerParam>> reshape_param_vec_ = {};
    std::vector<Blob *> reduce_inputs_                                 = {};
    std::vector<Blob *> reduce_outputs_                                = {};
    std::shared_ptr<Blob> reduce_input_blob_                           = nullptr;
    std::shared_ptr<Blob> reduce_output_blob_                          = nullptr;
    int single_axis_                                                   = 1;
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
