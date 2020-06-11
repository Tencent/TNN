// Copyright 2019 Tencent. All Rights Reserved

#ifndef TNN_SOURCE_DEVICE_CUDA_CUDA_SOFTMAX_LAYER_ACC_H_
#define TNN_SOURCE_DEVICE_CUDA_CUDA_SOFTMAX_LAYER_ACC_H_

#include <vector>

#include "device/cuda/acc/cuda_layer_acc.h"

namespace TNN_NS {

// @brief batch norm layer cuda acc
class CudaSoftmaxLayerAcc : public CudaLayerAcc {
public:


    /**
     * @brief init layer with param, resouce, intput blobs and output blobs.
     * @param param    layer param
     * @param resouce  layer resouce
     * @param inputs    input blobs
     * @param outputs   output blobs
     */
    virtual Status Init(Context *context, LayerParam *param,
                            LayerResource *            resource,
                            const std::vector<Blob *> &inputs,
                            const std::vector<Blob *> &outputs) override;

    // @brief virtual destrcutor
    virtual ~CudaSoftmaxLayerAcc() override;

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

    int axis_;
    int inner_dim_;
    int outer_dim_;
    float * workspace_;
    size_t  workspace_size_;

};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_DEVICE_CUDA_CUDA_SOFTMAX_LAYER_ACC_H_
