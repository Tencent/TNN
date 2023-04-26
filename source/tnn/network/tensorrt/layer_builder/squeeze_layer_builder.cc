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

#include "tnn/network/tensorrt/layer_builder/tensorrt_layer_builder.h"

namespace TNN_NS {

DECLARE_TENSORRT_LAYER_BUILDER(Squeeze, LAYER_SQUEEZE);

ILayer* SqueezeTRTLayerBuilder::AddToNetwork(INetworkDefinition* network) {
    auto paramlist = dynamic_cast<SqueezeLayerParam*>(param_);
    auto axes = paramlist->axes;
    auto tensor = GetInputITensors()[0];
    int size = tensor->getDimensions().nbDims;
    if (axes.empty()) {
        // TORCH has squeeze without dim,
        // it squeezes all dims != 1
        // This squeeze is dangerous, it is not encouraged, model trainers should make sure
        // that min_dim[i]==1, max_dim[i]!=1 cases should not happen, otherwise ERRORS may occur.
        DimsVector blob_dims = input_blobs_[0]->GetBlobDesc().dims;
        if (!blob_dims.empty()) {
            // We have input blob dim infomation,
            // This infomation is rather relible
            for (int i=0; i<blob_dims.size(); i++) {
                if (blob_dims[i] == 1) {
                    axes.push_back(i);
                } 
            }
        } else {
            // No input blob info available,
            // we use TRT ITensor dim
            // less reliable because dim[i] == -1 is not counted.
            LOGI("WARNING: Run into Squeeze TRT LayerBuilder with param->axes empty and input blob info EMPTY, axes now depends on TRT ITensor dim, may lead to potential error. torch.Squeeze(%in) with no dims is overall not recommended.");
            auto itensor_dims = tensor->getDimensions();
            for (int i=0; i<itensor_dims.nbDims; i++) {
                if (itensor_dims.d[i] == 1) {
                    axes.push_back(i);
                }
            }
        }
        if (!axes.empty()) {
            paramlist->axes = axes;
        }
    } else {
        for (auto& axis : axes) {
            if (axis < 0) {
                axis += size;
            }
        }
    }

    // Return ERROR if axes is still empty
    if (axes.empty()) {
        LOGE("SqueezeTRTLayerBuilder: Unable to to get or determine AXEs for Squeeze Layer.");
        return nullptr;
    }

    auto layer = addSqueeze(network, *tensor, axes);
    if (layer != nullptr) {
        layer->setName(layer_name_.c_str());
    }

    return layer;
}

REGISTER_TENSORRT_LAYER_BUILDER(Squeeze, LAYER_SQUEEZE);

}  //  namespace TNN_NS

