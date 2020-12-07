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

#include "tnn/core/abstract_layer_acc.h"
#include "tnn/core/profile.h"

#include <algorithm>

namespace TNN_NS {

Status AbstractLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                              const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    /*
     * Check whether the format is supported by LayerAcc or not.
     * The supported format of each layer is given by LayerAcc.
     */
    for (auto blob : outputs) {
        Status ret = ResolveBlobDataFormat(blob);
        if (ret != TNN_OK) {
            return ret;
        }
    }

    for (auto blob : inputs) {
        Status ret = ResolveBlobDataFormat(blob);
        if (ret != TNN_OK) {
            return ret;
        }
    }
    return TNN_OK;
}

#if TNN_PROFILE
void AbstractLayerAcc::UpdateProfilingData(ProfilingData *pdata, LayerParam *param, DimsVector input_dim,
                                           DimsVector output_dim) {
    if (!pdata) {
        return;
    }

    if (param) {
        pdata->op_name    = param->type;
        pdata->layer_name = param->name;
    }

    if (input_dim.size() > 0) {
        pdata->input_dims = input_dim;
    }
    if (output_dim.size() > 0) {
        pdata->output_dims = output_dim;
    }

    pdata->flops     = GetFlops();
    pdata->bandwidth = GetBandwidth();

    // for conv/deconv
    {
        auto conv_param = dynamic_cast<ConvLayerParam *>(param);
        if (conv_param) {
            pdata->kernel_shape.push_back(conv_param->output_channel);
            pdata->kernel_shape.push_back(conv_param->input_channel);
            pdata->kernel_shape.push_back(conv_param->kernels[1]);
            pdata->kernel_shape.push_back(conv_param->kernels[0]);
            pdata->stride_shape.push_back(conv_param->strides[1]);
            pdata->stride_shape.push_back(conv_param->strides[0]);
            pdata->pad_shape.push_back(conv_param->pads[2]);
            pdata->pad_shape.push_back(conv_param->pads[0]);
            pdata->dilation_shape.push_back(conv_param->dialations[1]);
            pdata->dilation_shape.push_back(conv_param->dialations[0]);
            pdata->group = conv_param->group;
        }
    }

    // for pool
    {
        auto pool_param = dynamic_cast<PoolingLayerParam *>(param);
        if (pool_param) {
            pdata->kernel_shape.push_back(pool_param->kernels[1]);
            pdata->kernel_shape.push_back(pool_param->kernels[0]);
            pdata->stride_shape.push_back(pool_param->strides[1]);
            pdata->stride_shape.push_back(pool_param->strides[0]);
            pdata->pad_shape.push_back(pool_param->pads[2]);
            pdata->pad_shape.push_back(pool_param->pads[0]);
        }
    }
}

double AbstractLayerAcc::GetFlops() {
    return 0;
}

double AbstractLayerAcc::GetBandwidth() {
    return 0;
}
#endif

Status AbstractLayerAcc::ResolveBlobDataFormat(Blob *blob) {
    BlobDesc desc                        = blob->GetBlobDesc();
    std::vector<DataFormat> support_list = SupportDataFormat(desc.data_type, static_cast<int>(desc.dims.size()));
    ASSERT(support_list.size() > 0);

    /*
     * DATA_FORMAT_AUTO : first format supported by the LayerAcc
     * Others:  return error if LayerAcc not support.
     */
    if (desc.data_format == DATA_FORMAT_AUTO) {
        desc.data_format = support_list[0];
        blob->SetBlobDesc(desc);
        return TNN_OK;
    } else {
        auto iter = std::find(support_list.begin(), support_list.end(), desc.data_format);
        if (iter != support_list.end()) {
            return TNN_OK;
        } else {
            return Status(TNNERR_DEVICE_ACC_DATA_FORMAT_NOT_SUPPORT, "unsupported data format for device acc");
        }
    }
}

}  // namespace TNN_NS
