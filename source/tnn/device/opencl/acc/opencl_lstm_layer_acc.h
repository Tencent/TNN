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

#ifndef TNN_SOURCE_TNN_DEVICE_OPENCL_ACC_OPENCL_LSTM_LAYER_ACC_H_
#define TNN_SOURCE_TNN_DEVICE_OPENCL_ACC_OPENCL_LSTM_LAYER_ACC_H_

#include "tnn/device/opencl/acc/opencl_layer_acc.h"
#include "tnn/device/opencl/opencl_memory.h"

namespace TNN_NS {
class OpenCLLSTMONNXLayerAcc : public OpenCLLayerAcc {
public:
    virtual Status Init(Context *context, LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                        const std::vector<Blob *> &outputs) override;

    virtual ~OpenCLLSTMONNXLayerAcc() override;

    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;

    virtual Status ReloadConstantBlobs(const std::vector<Blob *> &inputs, bool only_reload_shape_differ_blob = false) override;

private:
    virtual std::vector<DataFormat> SupportDataFormat(DataType data_type, int dims_size, BlobType blob_type) override;
    Status ConvertWeights(std::shared_ptr<RawBuffer> buffer, std::shared_ptr<Blob>& blob);
    Status ConvertBias(std::shared_ptr<RawBuffer> buffer, std::shared_ptr<Blob>& blob);
    Status ConvertInitialState(std::shared_ptr<RawBuffer> buffer, std::shared_ptr<Blob>& blob);
    Status CreateDefaultState(int num_directions, int batch_size, int hidden_size, std::shared_ptr<Blob>& blob);
    Status AllocateTempBlob(int num_directions, int hidden_size, int batch, int sequence, std::shared_ptr<Blob>& blob);

private:
    std::shared_ptr<Blob> ocl_gates_;
    std::shared_ptr<Blob> ocl_temp_out_;
    std::shared_ptr<Blob> ocl_zero_state_blob_; // default state blob
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_OPENCL_ACC_OPENCL_LSTM_LAYER_ACC_H_
