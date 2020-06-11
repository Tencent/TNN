// Copyright 2019 Tencent. All Rights Reserved

#ifndef TNN_SOURCE_DEVICE_CUDA_CUDA_CONTEXT_H_
#define TNN_SOURCE_DEVICE_CUDA_CUDA_CONTEXT_H_

#include <string>
#include <vector>

#include "cublas_v2.h"
#include "cudnn.h"

#include "core/context.h"

namespace TNN_NS {

class CudaContext : public Context {
public:
    /**
     * @brief deconstruct the cudacontext
     */
    ~CudaContext();

    // @brief get tnn command queue
    // @param command_queue device command queue for forward
    virtual Status GetCommandQueue(void** command_queue) override;

    /**
     * @brief init context with specified device id
     * @param device id
     */
    virtual Status Init(int device_id);

    /**
     * @brief deinit context
     */
    virtual Status DeInit();

    // load library
    virtual Status LoadLibrary(std::vector<std::string> path);

    /**
     * @brief befor instace forword
     * @param instance instace
     */
    virtual Status OnInstanceForwardBegin();
    /**
     * @brief after instace forword
     * @param instance instace
     */
    virtual Status OnInstanceForwardEnd();

    // @brief wait for jobs in the current context to complete
    virtual Status Synchronize() override;

public:
    cudnnHandle_t cudnn_handle_;
    cublasHandle_t cublas_handle_;
    cudaStream_t stream_;

    int device_id_;
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_DEVICE_CUDA_CUDA_CONTEXT_H_
