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

#ifndef TNN_SOURCE_TNN_DEVICE_CUDA_ACC_CUDA_LAYER_ACC_H_
#define TNN_SOURCE_TNN_DEVICE_CUDA_ACC_CUDA_LAYER_ACC_H_

#include <cuda_fp16.h>

#include "tnn/core/macro.h"
#include "tnn/device/cuda/cuda_context.h"
#include "tnn/device/cuda/cuda_device.h"
#include "tnn/device/cuda/cuda_macro.h"
#include "tnn/device/cuda/utils.cuh"

namespace TNN_NS {

struct CudaTempBufUnit {
    void* ptr = nullptr;
    uint32_t size;
};

class CudaLayerAcc : public AbstractLayerAcc {
public:
    /**
     * @brief init layer with param, resource, input blobs and output blobs.
     * @param param       layer param
     * @param resource    layer resource
     * @param inputs      input blobs
     * @param outputs     output blobs
     */
    virtual Status Init(Context *context, LayerParam *param, LayerResource *resource,
            const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

    // @brief virtual destrcutor
    virtual ~CudaLayerAcc();

    /**
     * @brief input or output blobs reshape.
     * @param inputs     input blobs
     * @param outputs    output blobs
     * @return reshape result
     */
    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

    /**
     * @brief layer forward
     * @param inputs     input blobs
     * @param outputs    output blobs
     * @return forward result
     */
    virtual Status Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

    // @brief allocate or update constant blobs if constant resource change。
    // Note: this func may cost much time, call this func only when necessary。
    virtual Status ReloadConstantBlobs(const std::vector<Blob *> &inputs);

protected:
    void CreateTempBuf(size_t size);

    CudaDevice *device_      = nullptr;
    LayerParam *param_       = nullptr;
    LayerResource *resource_ = nullptr;
    CudaContext *context_    = nullptr;
    std::vector<CudaTempBufUnit> tempbufs_;

private:
    // @brief retrun device layer acc support data format
    virtual std::vector<DataFormat> SupportDataFormat(DataType data_type, int dims_size);
};


#define DECLARE_CUDA_ACC_WITH_FUNC(type_string, layer_type, extra_funcs)                                              \
    class Cuda##type_string##LayerAcc : public CudaLayerAcc {                                                         \
    public:                                                                                                           \
        virtual Status Init(Context *context, LayerParam *param, LayerResource *resource,                             \
            const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);                                   \
        virtual ~Cuda##type_string##LayerAcc() {};                                                                    \
        virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);                \
        virtual Status Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);                \
        extra_funcs                                                                                                   \
    }

#define DECLARE_CUDA_ACC(type_string, layer_type)                                                                     \
    class Cuda##type_string##LayerAcc : public CudaLayerAcc {                                                         \
    public:                                                                                                           \
        virtual Status Init(Context *context, LayerParam *param, LayerResource *resource,                             \
            const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);                                   \
        virtual ~Cuda##type_string##LayerAcc() {};                                                                    \
        virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);                \
        virtual Status Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);                \
    }

#define REGISTER_CUDA_ACC(type_string, layer_type)                                                                    \
    CudaTypeLayerAccRegister<TypeLayerAccCreator<Cuda##type_string##LayerAcc>> g_cuda_##layer_type##_acc_register(    \
        layer_type);

}  //  namespace TNN_NS

#endif  //  TNN_SOURCE_TNN_DEVICE_CUDA_ACC_CUDA_LAYER_ACC_H_
