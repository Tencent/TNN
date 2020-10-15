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

#ifndef TNN_SOURCE_TNN_DEVICE_OPENCL_ACC_OPENCL_CPU_ADAPTER_H_
#define TNN_SOURCE_TNN_DEVICE_OPENCL_ACC_OPENCL_CPU_ADAPTER_H_

#include "tnn/core/abstract_layer_acc.h"
#include "tnn/core/context.h"
#include "tnn/device/opencl/opencl_context.h"

namespace TNN_NS {

class OpenCLCpuAdapterAcc : public AbstractLayerAcc {
public:

    OpenCLCpuAdapterAcc(LayerType impl_layer_type);

    virtual Status Init(Context *context, LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                        const std::vector<Blob *> &outputs) override;

    virtual ~OpenCLCpuAdapterAcc() override;

    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;

    virtual Status Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;

private:
    // @brief return device layer acc support data format
    virtual std::vector<DataFormat> SupportDataFormat(DataType data_type, int dims_size) override;

private:
    LayerType impl_layer_type_;
    
    DeviceType impl_device_type_;
    Context* impl_device_context_;
    AbstractLayerAcc* cpu_adapter_acc_ = nullptr;

    OpenCLContext *ocl_context_ = nullptr;

    std::vector<Blob *> cpu_blob_in_;
    std::vector<Blob *> cpu_blob_out_;
};

}

#endif  // TNN_SOURCE_TNN_DEVICE_OPENCL_ACC_OPENCL_CPU_ADAPTER_H_
