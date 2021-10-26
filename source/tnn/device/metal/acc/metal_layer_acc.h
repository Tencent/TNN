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

#ifndef TNN_SOURCE_TNN_DEVICE_METAL_METAL_LAYER_ACC_H_
#define TNN_SOURCE_TNN_DEVICE_METAL_METAL_LAYER_ACC_H_

#include "tnn/core/abstract_layer_acc.h"
#include "tnn/device/metal/acc/metal_common.h"
#include "tnn/device/metal/metal_device.h"
#include "tnn/device/metal/metal_macro.h"
#include "tnn/utils/blob_transfer_utils.h"

TNN_OBJC_CLASS(TNNMMetalContextImpl);

namespace TNN_NS {
class MetalContext;

// @brief conv layer metal acc
class MetalLayerAcc : public AbstractLayerAcc {
public:
    virtual Status Init(Context *context, LayerParam *param, LayerResource *resource,
                const std::vector<Blob *> &inputs,
                const std::vector<Blob *> &outputs);

    virtual ~MetalLayerAcc();

    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

    virtual Status AllocateBufferParam(const std::vector<Blob *> &inputs,
                                       const std::vector<Blob *> &outputs);

    virtual Status Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

    /**
     * @brief allocate or update constant blobs if constant resource changes.
     * Note: this func may cost much time, call this func only when necessary.
     */
    virtual Status ReloadConstantBlobs(const std::vector<Blob *> &inputs, bool only_reload_shape_differ_blob = false);


public:
    virtual std::string KernelName(const std::vector<Blob *> &inputs,
                                   const std::vector<Blob *> &outputs);
    
    virtual Status ComputeThreadSize(const std::vector<Blob *> &inputs,
                                     const std::vector<Blob *> &outputs,
                                     MTLSize &size);
    virtual Status ComputeThreadgroupSize(const std::vector<Blob *> &inputs,
                                     const std::vector<Blob *> &outputs,
                                     MTLSize &size);
    virtual Status SetKernelEncoderParam(id<MTLComputeCommandEncoder> encoder,
                                         const std::vector<Blob *> &inputs,
                                         const std::vector<Blob *> &outputs);

    virtual Status UpdateBlobDataType(const std::vector<Blob *> &inputs,
                                   const std::vector<Blob *> &outputs);

protected:
    LayerParam *param_       = nullptr;
    LayerResource *resource_ = nullptr;

    MetalContext *context_ = nullptr;

    id<MTLBuffer> buffer_param_ = nil;
    
    NSString *kernel_label_ = nil;
    NSString * GetKernelLabel();

    // @brief if true, const blobs are loaded by the naive device
    virtual bool UseNaiveConstantBlobs();
    // @brief config blobdesc for reloading buffer to metal blob
    virtual Status ConfigBuffer2MetalBlobDesc(BlobDesc &desc);
    // @brief reload buffer to metal blob
    virtual Status RawBuffer2MetalBlob(RawBuffer *buffer, std::shared_ptr<Blob> &blob, BlobDesc &desc);

private:
    virtual std::vector<DataFormat> SupportDataFormat(DataType data_type, int dims_size, BlobType blob_type);
};

MTLSize GetDefaultThreadSize(DimsVector dims, bool combineHeightWidth);

MTLSize GetDefaultThreadSizeFusedLast(DimsVector dims, bool combineHeightWidth);

struct MetalParams GetDefaultMetalParams(DimsVector input, DimsVector output);

// @brief allocate metal buffer form RawBuffer, like conv bias
// @context tnn instance device context
// @param buffer    input raw buffer
// @param count  the count of elements in RawBuffer
// @param status   output status
id<MTLBuffer> AllocateMetalBufferFormRawBuffer1D(RawBuffer buffer, int count, Status &status);

// @brief allocate packed metal buffer with format GOIHW4 form RawBuffer, like conv weight(gic or gic is not 4x)
// @context tnn instance device context
// @param buffer    input raw buffer
// @param buffer_shape  format OIHW
// @param group    group
// @param status   output status
// @param status   transpose transpose weght for deconv
id<MTLBuffer> AllocatePackedGOIHW4MetalBufferFormRawBuffer(RawBuffer buffer, DimsVector buffer_shape, int group,
                                                            Status &status, bool transpose = false);

// @brief allocate packed metal buffer with format GOIHW16 form RawBuffer, like conv weight
// @context tnn instance device context
// @param buffer    input raw buffer
// @param buffer_shape  format OIHW
// @param group    group
// @param status   output status
// @param status   transpose transpose weght for deconv
id<MTLBuffer> AllocatePackedGOIHW16MetalBufferFormRawBuffer(RawBuffer buffer, DimsVector buffer_shape, int group,
                                                            Status &status, bool transpose = false);

// @brief allocate packed metal buffer with format NC4HW4 form RawBuffer, like conv weight
// @context tnn instance device context
// @param buffer    input raw buffer
// @param buffer_shape  format OIHW
// @param group    group
// @param status   output status
id<MTLBuffer> AllocatePackedNC4HW4MetalBufferFormRawBuffer(RawBuffer buffer, DimsVector buffer_shape, int group,
                                                           Status &status);

void GetSingleAxisSplitSize(const DimsVector& dims, int axis, MTLSize& size, bool reduce_on_axis);

#define DECLARE_METAL_ACC(type_string, layer_type)                                                                     \
    class Metal##type_string##LayerAcc : public MetalLayerAcc {                                                        \
    public:                                                                                                            \
        virtual ~Metal##type_string##LayerAcc(){};                                                                     \
        virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);                 \
        virtual Status AllocateBufferParam(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);     \
        virtual Status Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);                 \
        virtual std::string KernelName(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs); \
        virtual Status ComputeThreadSize(const std::vector<Blob *> &inputs, \
                                 const std::vector<Blob *> &outputs, \
                                 MTLSize &size); \
        virtual Status SetKernelEncoderParam(id<MTLComputeCommandEncoder> encoder, \
                                     const std::vector<Blob *> &inputs, \
                                     const std::vector<Blob *> &outputs);\
    }

#define DECLARE_METAL_ACC_WITH_EXTRA(type_string, layer_type, extra)                                            \
    class Metal##type_string##LayerAcc : public MetalLayerAcc {                                                        \
    public:                                                                                                            \
        virtual ~Metal##type_string##LayerAcc(){};                                                                     \
        virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);                 \
        virtual Status AllocateBufferParam(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);     \
        virtual Status Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);                 \
        virtual std::string KernelName(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs); \
        virtual Status ComputeThreadSize(const std::vector<Blob *> &inputs, \
                                 const std::vector<Blob *> &outputs, \
                                 MTLSize &size); \
        virtual Status SetKernelEncoderParam(id<MTLComputeCommandEncoder> encoder, \
                                     const std::vector<Blob *> &inputs, \
                                     const std::vector<Blob *> &outputs);\
        extra; \
    }

#define REGISTER_METAL_ACC(type_string, layer_type)                                                                    \
    MetalTypeLayerAccRegister<TypeLayerAccCreator<Metal##type_string##LayerAcc>> g_metal_##layer_type##_acc_register(  \
        layer_type);

class MetalTypeLayerLayoutCreator {
public:
    static std::shared_ptr<ImplementedLayout> UpdateImplementedLayout(LayerType layer_type, DataFormat layout) {
        // make sure arm device has been registered
        TypeDeviceRegister<MetalDevice> metal_device_register(DEVICE_METAL);
        auto implemented_layout = GetDevice(DEVICE_METAL)->GetImplementedLayout(layer_type);
        auto updated_layout     = std::make_shared<ImplementedLayout>(*implemented_layout);
        updated_layout->layouts.push_back(layout);
        return updated_layout;
    }
};

#define REGISTER_METAL_LAYOUT(layer_type, layout)                                                                        \
    MetalTypeLayerLayoutRegister g_metal_##layer_type##_##layout##_layout_register(                                      \
             layer_type, MetalTypeLayerLayoutCreator::UpdateImplementedLayout(layer_type, layout));

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_METAL_METAL_LAYER_ACC_H_
