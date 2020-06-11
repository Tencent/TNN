// Copyright 2019 Tencent. All Rights Reserved

#ifndef TNN_SOURCE_DEVICE_CUDA_CUDA_CONV_3D_LAYER_ACC_H_
#define TNN_SOURCE_DEVICE_CUDA_CUDA_CONV_3D_LAYER_ACC_H_

#include <vector>

#include "device/cuda/acc/cuda_layer_acc.h"

namespace TNN_NS {

// @brief conv 3d layer cuda acc
class CudaConv3DLayerAcc : public CudaLayerAcc {
public:
    virtual Status Init(Context *context, LayerParam *param,
                        LayerResource *resource,
                        const std::vector<Blob *> &inputs,
                        const std::vector<Blob *> &outputs) override;

    virtual ~CudaConv3DLayerAcc() override;

    virtual Status Reshape(const std::vector<Blob *> &inputs,
                           const std::vector<Blob *> &outputs) override;

    virtual Status Forward(const std::vector<Blob *> &inputs,
                           const std::vector<Blob *> &outputs) override;

protected:
    CudaLayerKernelInfo kernel_;

    bool bias_term_;

    // cudnn descs
    bool descs_setup_;
    bool algo_inited_;

    cudnnTensorFormat_t tensor_format_;
    cudnnDataType_t data_type_;

    cudnnConvolutionMode_t conv_mode_;
    cudnnConvolutionFwdAlgo_t conv_algo_;

    cudnnTensorDescriptor_t bottom_desc_;
    cudnnTensorDescriptor_t top_desc_;

    cudnnTensorDescriptor_t bias_desc_;
    cudnnFilterDescriptor_t filter_desc_;
    cudnnConvolutionDescriptor_t conv_desc_;

    float alpha_;
    float beta_;

    bool workspace_setup_;
    size_t workspace_size_;

    float *weights_;
    float *bias_;
    float *workspace_data_;
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_DEVICE_CUDA_CUDA_CONV_3D_LAYER_ACC_H_
