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

#ifndef TNN_SOURCE_TNN_DEVICE_OPENCL_OPENCL_BLOB_CONVERTER_H_
#define TNN_SOURCE_TNN_DEVICE_OPENCL_OPENCL_BLOB_CONVERTER_H_

#include "tnn/core/macro.h"
#include "tnn/device/opencl/opencl_utils.h"
#include "tnn/utils/blob_converter.h"
#include "tnn/utils/blob_converter_internal.h"

namespace TNN_NS {

class OpenCLBlobConverterAcc : public BlobConverterAcc {
public:
    OpenCLBlobConverterAcc(Blob* blob);
    virtual ~OpenCLBlobConverterAcc();

    virtual Status ConvertToMat(Mat& image, MatConvertParam param, void* command_queue = NULL);
    virtual Status ConvertToMatAsync(Mat& image, MatConvertParam param, void* command_queue = NULL);

    virtual Status ConvertFromMat(Mat& image, MatConvertParam param, void* command_queue = NULL);
    virtual Status ConvertFromMatAsync(Mat& image, MatConvertParam param, void* command_queue = NULL);

private:
    Status CreateConvertUnit(OpenCLExecuteUnit& unit, Mat& mat, MatConvertParam param, bool convert_to_mat);
    void CalculateWorkgroupSize(OpenCLExecuteUnit& unit);
    Status SetConvertArgs(OpenCLExecuteUnit& unit, Mat& mat, MatConvertParam param, bool convert_to_mat);

    Status RunConvertUnit(OpenCLExecuteUnit& unit, cl::CommandQueue* command_queue, bool need_wait = false);
    Status CopyBufferDataToMat(Mat& mat, cl::CommandQueue* command_queue);
    Status CopyMatToBufferData(Mat& mat, cl::CommandQueue* command_queue);
    Status CopyScaleBiasToBuffer(MatConvertParam param, cl::CommandQueue *cl_command_queue);

    Status GetConvertToMatKernelName(Mat &mat, std::string& kernel_name, std::string& program_name);
    Status GetConvertFromMatKernelName(Mat &mat, std::string& kernel_name, std::string& program_name);

    std::map<std::string, OpenCLExecuteUnit> convert_to_mat_map_ = {};
    std::map<std::string, OpenCLExecuteUnit> convert_from_mat_map_ = {};
    std::shared_ptr<cl::Buffer> buffer_ = nullptr;
    std::shared_ptr<cl::Buffer> scale_buffer_ = nullptr;
    std::shared_ptr<cl::Buffer> bias_buffer_ = nullptr;
    std::vector<float> host_scale_buffer_;
    std::vector<float> host_bias_buffer_;
    int64_t buffer_size_ = 0;
    int scale_bias_buffer_size_ = 0;
    bool do_scale_bias_ = true;
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_OPENCL_OPENCL_BLOB_CONVERTER_H_
