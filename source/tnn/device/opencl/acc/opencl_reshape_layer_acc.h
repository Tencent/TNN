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

#ifndef TNN_SOURCE_TNN_DEVICE_OPENCL_ACC_OPENCL_RESHAPE_LAYER_ACC_H_
#define TNN_SOURCE_TNN_DEVICE_OPENCL_ACC_OPENCL_RESHAPE_LAYER_ACC_H_

#include "tnn/device/opencl/acc/opencl_layer_acc.h"

namespace TNN_NS {

class OpenCLReshapeLayerAcc : public OpenCLLayerAcc {
public:
    virtual Status Init(Context *context, LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                        const std::vector<Blob *> &outputs) override;

    virtual ~OpenCLReshapeLayerAcc() override;

    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;

private:
    virtual std::vector<DataFormat> SupportDataFormat(DataType data_type, int dims_size, BlobType blob_type) override;
    virtual std::vector<DataType> SupportDataType(int dims_size, BlobType blob_type) override;
    std::shared_ptr<cl::Buffer> inter_buffer_ = nullptr;
    int input_dims_size_ = 0;
    int output_dims_size_ = 0;
    bool is_nchw_output_ = false;
    std::string im_to_bf_func_name_;
    std::string bf_to_im_func_name_;
    std::string im_to_bf_program_name_;
    std::string bf_to_im_program_name_;
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_OPENCL_ACC_OPENCL_RESHAPE_LAYER_ACC_H_
