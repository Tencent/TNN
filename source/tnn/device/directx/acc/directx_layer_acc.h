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

#ifndef TNN_SOURCE_TNN_DEVICE_DIRECTX_ACC_DIRECTX_LAYER_ACC_H_
#define TNN_SOURCE_TNN_DEVICE_DIRECTX_ACC_DIRECTX_LAYER_ACC_H_

#include <vector>
#include <memory>

#include "tnn/core/abstract_layer_acc.h"

#include "tnn/device/directx/directx_context.h"
#include "tnn/device/directx/directx_device.h"
// #include "tnn/device/directx/directx_execute_unit.h"
#include "tnn/device/directx/directx_runtime.h"
#include "tnn/device/directx/directx_util.h"
#include "tnn/device/directx/directx_common.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

namespace directx {

class DirectXLayerAcc : public AbstractLayerAcc {
public:
    virtual Status Init(Context *context, LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                        const std::vector<Blob *> &outputs) override;

    virtual ~DirectXLayerAcc() override;

    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;

    virtual Status Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;

    virtual Status DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) = 0;

    virtual Status ReloadConstantBlobs(const std::vector<Blob *> &inputs, bool only_reload_shape_differ_blob = false) override;

protected:

    void ConfigKernelStrategy();

    // Status ConvertChannelWeights(RawBuffer &raw_handle, shared_ptr<DirectXMemory> &ocl_handle, int output_channel,
    //                              bool has_value = true, bool share_channel = false, bool use_buffer = false);

    Status RawBuffer2DirectXBlob(RawBuffer *buffer, std::shared_ptr<Blob> &blob, DataFormat format = DATA_FORMAT_NCHW);

    // @brief Get the ID3DDeviceContext
    std::shared_ptr<ID3D11DeviceContext> GetID3DContext();

    // @brief Get the ID3DDevice
    std::shared_ptr<ID3D11Device> GetID3DDevice();

    DirectXContext *dx_context_ = nullptr;
    // std::vector<OpenCLExecuteUnit> execute_units_ = {};

    LayerParam *param_ = nullptr;
    LayerResource *resource_ = nullptr;
    std::string layer_name_ = "";
    DimsVector input_dims_ = {};
    DimsVector output_dims_ = {};

    DataType data_type_;

    GpuInfo gpu_info_;
    bool use_buffer_     = false;

#if TNN_PROFILE
    std::shared_ptr<DirectXProfilingData> profiling_data = nullptr;
#endif

private:
    // Status ConvertChannelWeights(float *handle_data_ptr, shared_ptr<DirectXMemory> &ocl_handle, int output_channel,
    //                              bool has_handle = true, bool share_channel = false, bool use_buffer = false);

    Status CheckBlob(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

    // @brief return device layer acc support data format
    virtual std::vector<DataFormat> SupportDataFormat(DataType data_type, int dims_size, BlobType blob_type) override;

    // @brief return device layer acc support data type
    virtual std::vector<DataType> SupportDataType(int dims_size, BlobType blob_type);

    // @brief decide Blob Data Type based on support data type list
    virtual Status ResolveBlobDataType(Blob *blob, BlobType blob_type);

};

#define DECLARE_DIRECTX_ACC(type_string)                                                                               \
    class DirectX##type_string##LayerAcc : public DirectXLayerAcc {                                                    \
    public:                                                                                                            \
        virtual ~DirectX##type_string##LayerAcc(){};                                                                   \
        virtual Status Init(Context *context, LayerParam *param, LayerResource *resource,                              \
                            const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;           \
        virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;        \
    }

#define REGISTER_DIRECTX_ACC(type_string, layer_type)                                                                  \
    DirectXLayerAccRegister<TypeLayerAccCreator<DirectX##type_string##LayerAcc>>                                       \
        g_directx_##layer_type##_acc_register(layer_type);

class DirectXTypeLayerLayoutCreator {
public:
    static std::shared_ptr<ImplementedLayout> UpdateImplementedLayout(LayerType layer_type, DataFormat layout) {
        // make sure directx device has been registered
        TypeDeviceRegister<DirectXDevice> directx_device_register(DEVICE_DIRECTX);
        auto implemented_layout = GetDevice(DEVICE_DIRECTX)->GetImplementedLayout(layer_type);
        auto updated_layout     = std::make_shared<ImplementedLayout>(*implemented_layout);
        updated_layout->layouts.push_back(layout);
        return updated_layout;
    }
};

#define REGISTER_DIRECTX_LAYOUT(layer_type, layout)                                                                    \
    DirectXLayerLayoutRegister g_directx_##layer_type##_##layout##_layout_register(                                    \
             layer_type, DirectXTypeLayerLayoutCreator::UpdateImplementedLayout(layer_type, layout));

} // namespace directx

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_DIRECTX_ACC_DIRECTX_LAYER_ACC_H_
