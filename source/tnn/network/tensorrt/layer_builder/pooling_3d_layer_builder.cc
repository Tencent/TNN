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

#include "tnn/network/tensorrt/layer_builder/tensorrt_plugin_layer_builder.h"

#include "tnn/network/tensorrt/utils.h"

namespace TNN_NS {

DECLARE_TENSORRT_LAYER_BUILDER(Pooling3D, LAYER_POOLING_3D);

ILayer* Pooling3DTRTLayerBuilder::AddToNetwork(INetworkDefinition* network) {
    auto paramlist = dynamic_cast<PoolingLayerParam*>(param_);
    auto input_foreign_tensor = dynamic_cast<ForeignBlob*>(input_blobs_[0])->GetForeignTensor();
    auto output_foreign_tensor = dynamic_cast<ForeignBlob*>(output_blobs_[0])->GetForeignTensor();
    auto input_tensor = std::dynamic_pointer_cast<TensorRTTensor>(input_foreign_tensor)->GetTensor();

    bool symmetric = (paramlist->pads[0] == paramlist->pads[1])
                  && (paramlist->pads[0] == paramlist->pads[2])
                  && (paramlist->pads[3] == paramlist->pads[4])
                  && (paramlist->pads[3] == paramlist->pads[5]);

    if (paramlist->is_global_pool) {
        ReduceOperation op;
        if (paramlist->pool_type == 0) {
            op = ReduceOperation::kMAX;
        } else {
            op = ReduceOperation::kAVG;
        }
        uint32_t reduceAxes = ((1 << input_tensor->getDimensions().nbDims) - 1) & ~0b11;

        ILayer* reduce = network->addReduce(*input_tensor, op, reduceAxes, true);
        reduce->setName(layer_name_.c_str());
        return reduce;
    }

    Dims kernelSize(ConvertToTRTDimsReverse(paramlist->kernels));

    PoolingType type;
    if (paramlist->pool_type == 0) {
        type = PoolingType::kMAX;
    } else {
        type = PoolingType::kAVERAGE;
    }

    IPoolingLayer *layer;
    auto pads = paramlist->pads;

    bool padNeg = false;
    for(const auto& p : pads)
        padNeg |= p < 0;

    if (padNeg) {
        DimsVector postPadding{pads[5], pads[3], pads[1]};
        DimsVector  prePadding{pads[4], pads[2], pads[0]};
        IPaddingLayer* padding_layer = network->addPaddingNd(*input_tensor,
                                                    ConvertToTRTDims(prePadding),
                                                    ConvertToTRTDims(postPadding));
        input_tensor = padding_layer->getOutput(0);
        pads = {0, 0, 0, 0, 0, 0};
    }
    layer = network->addPoolingNd(*input_tensor, type, kernelSize);
    if (layer != nullptr) {
        layer->setName(layer_name_.c_str());
        layer->setStrideNd(ConvertToTRTDimsReverse(paramlist->strides));
        if (!padNeg) {
            if (symmetric) {
                layer->setPaddingNd(ConvertPaddingToTRTDims(pads));
            } else {
                DimsVector postPadding{pads[5], pads[3], pads[1]};
                DimsVector  prePadding{pads[4], pads[2], pads[0]};
                layer->setPrePadding(ConvertToTRTDims(prePadding));
                layer->setPostPadding(ConvertToTRTDims(postPadding));
            }
        }
        if (paramlist->pad_type == -1) {
            if (paramlist->ceil_mode == 1) {
                layer->setPaddingMode(PaddingMode::kCAFFE_ROUND_UP);
            } else {
                layer->setPaddingMode(PaddingMode::kCAFFE_ROUND_DOWN);
            }
        } else if (paramlist->pad_type == 0) {
            layer->setPaddingMode(PaddingMode::kSAME_UPPER);
        } else if (paramlist->pad_type == 1) {
            layer->setPaddingMode(PaddingMode::kEXPLICIT_ROUND_UP);
        }
        if (paramlist->pool_type == 1) {
            layer->setAverageCountExcludesPadding(true);
        }
    }

    return layer;
}

REGISTER_TENSORRT_LAYER_BUILDER(Pooling3D, LAYER_POOLING_3D);

}  //  namespace TNN_NS