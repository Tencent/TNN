// Copyright 2019 Tencent. All Rights Reserved

#ifndef TNN_SOURCE_DEVICE_CUDA_CUDA_DEVICE_H_
#define TNN_SOURCE_DEVICE_CUDA_CUDA_DEVICE_H_

#include <memory.h>
#include "core/abstract_device.h"

namespace TNN_NS {

// @brief CudaDevice create cuda memory and cuda layer acc

class CudaDevice : public AbstractDevice {
public:
    explicit CudaDevice(DeviceType device_type);

    ~CudaDevice();

    virtual BlobMemorySizeInfo Calculate(BlobDesc& desc);

    virtual Status Allocate(void** handle, BlobMemorySizeInfo& size_info);

    virtual Status Free(void* handle);

    virtual Status CopyToDevice(BlobHandle* dst, const BlobHandle* src,
                                BlobDesc& desc, void* command_queue);

    virtual Status CopyFromDevice(BlobHandle* dst, const BlobHandle* src,
                                  BlobDesc& desc, void* command_queue);

    virtual AbstractLayerAcc* CreateLayerAcc(LayerType type);

    virtual Context* CreateContext(int device_id);

    static Status RegisterLayerAccCreator(
        LayerType type, std::shared_ptr<LayerAccCreator> creator);

private:
    static std::map<LayerType, std::shared_ptr<LayerAccCreator>>&
    GetLayerCreatorMap();
};

//@brief CudaTypeLayerAccRegister register CudaTypeLayerAccCreator
template <typename T>
class CudaTypeLayerAccRegister {
public:
    explicit CudaTypeLayerAccRegister(LayerType type) {
        CudaDevice::RegisterLayerAccCreator(type, std::make_shared<T>());
    }
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_DEVICE_CUDA_CUDA_DEVICE_H_
