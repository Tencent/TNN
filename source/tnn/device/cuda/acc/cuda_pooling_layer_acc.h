// Copyright 2019 Tencent. All Rights Reserved

#ifndef TNN_SOURCE_DEVICE_CUDA_CUDA_POOLING_3D_LAYER_ACC_H_
#define TNN_SOURCE_DEVICE_CUDA_CUDA_POOLING_3D_LAYER_ACC_H_

#include <vector>

#include "device/cuda/acc/cuda_layer_acc.h"

namespace TNN_NS {

// @brief conv 3d layer cuda acc
class CudaPoolingLayerAcc : public CudaLayerAcc {
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
    virtual ~CudaPoolingLayerAcc() override;

    /**
     * @brief input or output blobs reshape.
     * @param inputs    input blobs
     * @param outputs   output blobs
     * @return reshape result
     */
    virtual Status Reshape(const std::vector<Blob *> &inputs,
                           const std::vector<Blob *> &outputs) override;

    /**
     * @brief layer forward
     * @param inputs    input blobs
     * @param outputs   output blobs
     * @return execution result
     */
    virtual Status Forward(const std::vector<Blob *> &inputs,
                           const std::vector<Blob *> &outputs) override;

protected:
    CudaLayerKernelInfo kernel_;

    // cudnn descs
    bool descs_setup_;

    cudnnTensorFormat_t tensor_format_;
    cudnnDataType_t data_type_;

    cudnnPoolingMode_t pooling_mode_;
    cudnnPoolingDescriptor_t pool_desc_;

    cudnnTensorDescriptor_t bottom_desc_;
    cudnnTensorDescriptor_t top_desc_;

    float alpha_;
    float beta_;

    bool workspace_setup_;
    size_t workspace_size_;
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_DEVICE_CUDA_CUDA_POOLING_3D_LAYER_ACC_H_
