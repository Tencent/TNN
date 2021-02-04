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

#include "tnn/core/blob_int8.h"
#include "tnn/utils/naive_compute.h"
#include "tnn/device/cpu/acc/cpu_layer_acc.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

class CpuReformatLayerAcc : public CpuLayerAcc {
    // @brief virtual destrcutor
    virtual ~CpuReformatLayerAcc();

    /**
     * @brief init layer with param, resouce, intput blobs and output blobs.
     * @param context cpu context
     * @param param    layer param
     * @param resource  layer resouce
     * @param inputs    input blobs
     * @param outputs   output blobs
     */
    Status Init(Context *context, LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                const std::vector<Blob *> &outputs);

    /**
     * @brief input or output blobs reshape.
     * @param inputs    input blobs
     * @param outputs   output blobs
     * @return reshape result
     */
    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

    /**
     * @brief layer forward
     * @param inputs    input blobs
     * @param outputs   output blobs
     * @return execution result
     */
    virtual Status Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
};

CpuReformatLayerAcc::~CpuReformatLayerAcc() {}

Status CpuReformatLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                 const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto reformat_param = dynamic_cast<ReformatLayerParam *>(param);
    CHECK_PARAM_NULL(reformat_param);

    if (reformat_param->src_type == DATA_TYPE_INT8 && reformat_param->dst_type == DATA_TYPE_FLOAT) {
        reformat_param->type = DEQUANT_ONLY;
    } else if (reformat_param->src_type == DATA_TYPE_FLOAT && reformat_param->dst_type == DATA_TYPE_INT8) {
        reformat_param->type = QUANT_ONLY;
    } else {
        return Status(TNNERR_LAYER_ERR, "Error: cpu layer acc got unsupported data type.");
    }
    return CpuLayerAcc::Init(context, param, resource, inputs, outputs);
}

Status CpuReformatLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto reformat_param = dynamic_cast<ReformatLayerParam *>(param_);
    CHECK_PARAM_NULL(reformat_param);

    if (reformat_param->src_type == DATA_TYPE_INT8 && reformat_param->dst_type == DATA_TYPE_FLOAT) {
        reformat_param->type = DEQUANT_ONLY;
    } else if (reformat_param->src_type == DATA_TYPE_FLOAT && reformat_param->dst_type == DATA_TYPE_INT8) {
        reformat_param->type = QUANT_ONLY;
    } else {
        return Status(TNNERR_LAYER_ERR, "Error: cpu layer acc got unsupported data type.");
    }
    return TNN_OK;
}

Status CpuReformatLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto param = dynamic_cast<ReformatLayerParam *>(param_);
    CHECK_PARAM_NULL(param);
    auto dims       = outputs[0]->GetBlobDesc().dims;
    size_t datasize = DataTypeUtils::GetBytesSize(outputs[0]->GetBlobDesc().data_type);

    IntScaleResource *re;
    if (param->src_type == DATA_TYPE_INT8) {
        re = reinterpret_cast<BlobInt8 *>(inputs[0])->GetIntResource();
    } else if (param->dst_type == DATA_TYPE_INT8) {
        re = reinterpret_cast<BlobInt8 *>(outputs[0])->GetIntResource();
    } else {
        return Status(TNNERR_LAYER_ERR, "Error: cpu layer acc got unsupported data type.");
    }

    if (param->type == DEQUANT_ONLY) {
        NaiveDequant(reinterpret_cast<int8_t *>(inputs[0]->GetHandle().base), re->scale_handle.force_to<float *>(),
                    re->scale_handle.GetDataCount(), reinterpret_cast<float *>(outputs[0]->GetHandle().base), dims);
    } else if (param->type == QUANT_ONLY) {
        NaiveQuant(reinterpret_cast<float *>(inputs[0]->GetHandle().base), re->scale_handle.force_to<float *>(),
                  re->scale_handle.GetDataCount(), reinterpret_cast<int8_t *>(outputs[0]->GetHandle().base), dims);
    }
    return TNN_OK;
}

REGISTER_CPU_ACC(Reformat, LAYER_REFORMAT);

}  // namespace TNN_NS
