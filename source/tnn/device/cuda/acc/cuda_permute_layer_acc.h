// Copyright 2019 Tencent. All Rights Reserved

#ifndef TNN_SOURCE_DEVICE_CUDA_CUDA_PERMUTE_LAYER_ACC_H_
#define TNN_SOURCE_DEVICE_CUDA_CUDA_PERMUTE_LAYER_ACC_H_

#include <vector>

#include "device/cuda/acc/cuda_layer_acc.h"

namespace TNN_NS {

// @brief conv layer cuda acc
class CudaPermuteLayerAcc : public CudaLayerAcc {
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
    virtual ~CudaPermuteLayerAcc() override;

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
    int n_dims_;

    std::vector<int> permute_order_;
    std::vector<int> old_steps_;
    std::vector<int> new_steps_;

    int *permute_order_d_;
    int *old_steps_d_;
    int *new_steps_d_;
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_DEVICE_CUDA_CUDA_PERMUTE_LAYER_ACC_H_
