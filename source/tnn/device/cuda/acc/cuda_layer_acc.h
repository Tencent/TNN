// Copyright 2019 Tencent. All Rights Reserved

#ifndef TNN_SOURCE_DEVICE_CUDA_BASE_LAYER_ACC_H_
#define TNN_SOURCE_DEVICE_CUDA_BASE_LAYER_ACC_H_

#include "core/abstract_layer_acc.h"

#include <vector>

#include "cudnn.h"
#include "device/cuda/cuda_common.h"
#include "device/cuda/cuda_context.h"
#include "device/cuda/cuda_device.h"
#include "device/cuda/cuda_macro.h"

namespace TNN_NS {

// @brief cuda layer acc base type
class CudaLayerAcc : public AbstractLayerAcc {
public:
    /**
     * @brief init layer with param, resouce, intput blobs and output blobs.
     * @param param    layer param
     * @param resouce  layer resouce
     * @param inputs    input blobs
     * @param outputs   output blobs
     */
    virtual Status Init(Context *context, LayerParam *param,
                        LayerResource *resource,
                        const std::vector<Blob *> &inputs,
                        const std::vector<Blob *> &outputs) override;

    // @brief virtual destrcutor
    virtual ~CudaLayerAcc() override;

    /**
     * @brief input or output blobs reshape.
     * @param inputs    input blobs
     * @param outputs   output blobs
     * @return reshape result
     */
    virtual Status Reshape(const std::vector<Blob *> &inputs,
                           const std::vector<Blob *> &outputs) override = 0;

    /**
     * @brief layer forward
     * @param inputs    input blobs
     * @param outputs   output blobs
     * @return execution result
     */
    virtual Status Forward(const std::vector<Blob *> &inputs,
                           const std::vector<Blob *> &outputs) override = 0;

protected:
    template <class T>
    using vector = typename std::vector<T>;

    CudaContext *context_;
    LayerParam *param_;
    CudaLayerBlobInfo blob_info_;
    vector<CudaMemory> weights_;

private:
    // @brief return device layer acc support data format
    virtual std::vector<DataFormat> SupportDataFormat(DataType data_type,
                                                      int dims_size);
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_DEVICE_CUDA_CUDA_CONV_LAYER_ACC_H_
