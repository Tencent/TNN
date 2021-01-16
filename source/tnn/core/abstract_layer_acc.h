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

#ifndef TNN_SOURCE_TNN_CORE_LAYER_ACC_H_
#define TNN_SOURCE_TNN_CORE_LAYER_ACC_H_

#include <vector>

#include "tnn/core/blob.h"
#include "tnn/core/common.h"
#include "tnn/core/context.h"
#include "tnn/core/layer_type.h"
#include "tnn/core/profile.h"
#include "tnn/core/status.h"
#include "tnn/interpreter/layer_param.h"
#include "tnn/interpreter/layer_resource.h"
//#include "tnn/memory_manager/blob_memory_pool.h"

namespace TNN_NS {
class BlobMemoryPool;

// @brief AbstractLayerAcc define the layer acc interface
class AbstractLayerAcc {
public:
    // @brief virtual destructor
    virtual ~AbstractLayerAcc(){};

    // @brief init layer with param, resouce, intput blobs and output blobs.
    // @context tnn instance device context
    // @param param    layer param
    // @param resouce  layer resouce
    // @param inputs    input blobs
    // @param outputs   output blobs
    virtual Status Init(Context *context, LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                        const std::vector<Blob *> &outputs) = 0;

    // @brief prepare with inputs and outpus.
    // @param inputs    input blobs
    // @param outputs   output blobs
    // @return reshape result
    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) = 0;

    // @brief layer forward acc
    // @param inputs    input blobs
    // @param outputs   output blobs
    // @return execution result
    virtual Status Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) = 0;

    // @brief before layer acc forward
    // @param inputs    input blobs
    // @param outputs   output blobs
    // @return execution result
    virtual Status BeforeForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

    // @brief infer output blob shape in runtime where shape is determined by input data like constant of shape, it is different from layer::InferOutputShape where shape is determined by input shape and param.
    // @param inputs    input blobs
    // @param outputs   output blobs
    // @return execution result
    virtual Status InferRuntimeOutputShape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
    
    // @brief allocate output blob in runtime
    // @param inputs    input blobs
    // @param outputs   output blobs
    // @return execution result
    virtual Status AllocateRuntimeOutputBlob(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
    
    // @brief allocate or update constant blobs if constant resource change。
    // Note: this func may cost much time, call this func only when necessary。
    virtual Status ReloadConstantBlobs(const std::vector<Blob *> &inputs);
    
    // @brief after layer acc forward
    // @param inputs    input blobs
    // @param outputs   output blobs
    // @return execution result
    virtual Status AfterForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
    
    // @brief set runtime bolob pool
    void SetRuntimeBlobMemoryPool(BlobMemoryPool *runtime_blob_pool);
    
    // @brief set constant resource
    void SetConstantResource(ConstantResource* consts);
    
    // @brief set runtime mode
    void SetRuntimeMode(RuntimeMode mode);
    
#if TNN_PROFILE
    virtual void UpdateProfilingData(ProfilingData *pdata, LayerParam *param, DimsVector input_dim,
                                     DimsVector output_dim);
    virtual double GetFlops();
    virtual double GetBandwidth();
#endif

private:
    // @brief return device layer acc support data format
    virtual std::vector<DataFormat> SupportDataFormat(DataType data_type, int dims_size) = 0;

    // @brief decide Blob Data Format based on support data format list
    Status ResolveBlobDataFormat(Blob *blob);
    
protected:
    BlobMemoryPool *runtime_blob_pool_ = nullptr;
    ConstantResource* const_resource_ = nullptr;
    std::map<std::string, std::shared_ptr<Blob> > const_blob_map_ = {};
    RuntimeMode runtime_model_ = RUNTIME_MODE_NORMAL;
};

// @brief LayerAccCreator define create layer acc interface
class LayerAccCreator {
public:
    virtual AbstractLayerAcc *CreateLayerAcc(LayerType layer_type) = 0;
    virtual ~LayerAccCreator(){};
};

// @brief TypeLayerAccCreator template define type layer acc creator, different
// type layer acc instantiate template as parameter T.
template <typename T>
class TypeLayerAccCreator : public LayerAccCreator {
    AbstractLayerAcc *CreateLayerAcc(LayerType layer_type) {
        return new T();
    }
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_CORE_LAYER_ACC_H_
