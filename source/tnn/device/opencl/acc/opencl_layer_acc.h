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

#ifndef TNN_SOURCE_TNN_DEVICE_OPENCL_ACC_OPENCL_LAYER_ACC_H_
#define TNN_SOURCE_TNN_DEVICE_OPENCL_ACC_OPENCL_LAYER_ACC_H_

#include <vector>

#include "tnn/core/abstract_layer_acc.h"

#include "tnn/device/opencl/opencl_context.h"
#include "tnn/device/opencl/opencl_device.h"
#include "tnn/device/opencl/opencl_execute_unit.h"
#include "tnn/device/opencl/opencl_runtime.h"
#include "tnn/device/opencl/opencl_utils.h"

namespace TNN_NS {

class OpenCLLayerAcc : public AbstractLayerAcc {
public:
    virtual Status Init(Context *context, LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                        const std::vector<Blob *> &outputs) override;

    virtual ~OpenCLLayerAcc() override;

    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;

    virtual Status Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;

    virtual Status ReloadConstantBlobs(const std::vector<Blob *> &inputs, bool only_reload_shape_differ_blob = false) override;

#if TNN_PROFILE
    virtual void UpdateProfilingData(OpenCLProfilingData *pdata, std::vector<uint32_t> gws, std::vector<uint32_t> lws,
                                     int idx = 0);
#endif
protected:
    virtual bool NeedFlush();

    void ConfigKernelStrategy();

    void InsertUnactiveUnitId(int id);

    Status ConvertChannelWeights(RawBuffer &raw_handle, shared_ptr<OpenCLMemory> &ocl_handle, int output_channel,
                                 bool has_value = true, bool share_channel = false, bool use_buffer = false);

    Status RawBuffer2OpenCLBlob(RawBuffer *buffer, std::shared_ptr<Blob> &blob, DataFormat format = DATA_FORMAT_NHC4W4);
    OpenCLContext *ocl_context_ = nullptr;
    std::vector<OpenCLExecuteUnit> execute_units_ = {};

    LayerParam *param_ = nullptr;
    LayerResource *resource_ = nullptr;
    std::string op_name_ = "";
    std::string layer_name_ = "";
    DimsVector input_dims_ = {};
    DimsVector output_dims_ = {};
    std::set<std::string> build_options_ = {};

    GpuInfo gpu_info_;
    bool run_3d_ndrange_ = false;
    bool use_buffer_     = false;

private:
    Status ConvertChannelWeights(float *handle_data_ptr, shared_ptr<OpenCLMemory> &ocl_handle, int output_channel,
                                 bool has_handle = true, bool share_channel = false, bool use_buffer = false);

    Status CheckBlob(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
    // @brief return device layer acc support data format
    virtual std::vector<DataFormat> SupportDataFormat(DataType data_type, int dims_size, BlobType blob_type) override;

    // @brief return device layer acc support data type
    virtual std::vector<DataType> SupportDataType(int dims_size, BlobType blob_type);

    // @brief decide Blob Data Type based on support data type list
    virtual Status ResolveBlobDataType(Blob *blob, BlobType blob_type);

    std::set<int> unactive_unit_ids_ = {};
};

#define DECLARE_OPENCL_ACC(type_string)                                                                                \
    class OpenCL##type_string##LayerAcc : public OpenCLLayerAcc {                                                      \
    public:                                                                                                            \
        virtual ~OpenCL##type_string##LayerAcc(){};                                                                    \
        virtual Status Init(Context *context, LayerParam *param, LayerResource *resource,                              \
                            const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;           \
        virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;        \
    }

#define REGISTER_OPENCL_ACC(type_string, layer_type)                                                                   \
    OpenCLTypeLayerAccRegister<TypeLayerAccCreator<OpenCL##type_string##LayerAcc>>                                     \
        g_opencl_##layer_type##_acc_register(layer_type);

class OpenCLTypeLayerLayoutCreator {
public:
    static std::shared_ptr<ImplementedLayout> UpdateImplementedLayout(LayerType layer_type, DataFormat layout) {
        // make sure opencl device has been registered
        TypeDeviceRegister<OpenCLDevice> opencl_device_register(DEVICE_OPENCL);
        auto implemented_layout = GetDevice(DEVICE_OPENCL)->GetImplementedLayout(layer_type);
        auto updated_layout     = std::make_shared<ImplementedLayout>(*implemented_layout);
        updated_layout->layouts.push_back(layout);
        return updated_layout;
    }
};

#define REGISTER_OPENCL_LAYOUT(layer_type, layout)                                                                        \
    OpenCLTypeLayerLayoutRegister g_opencl_##layer_type##_##layout##_layout_register(                                      \
             layer_type, OpenCLTypeLayerLayoutCreator::UpdateImplementedLayout(layer_type, layout));

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_OPENCL_ACC_OPENCL_LAYER_ACC_H_
