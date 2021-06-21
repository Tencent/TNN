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

#include <cassert>
#include <stdlib.h>
#include <unordered_set>

#include "tnn/network/tensorrt/layer_builder/tensorrt_layer_builder.h"

namespace TNN_NS {

DECLARE_TENSORRT_LAYER_BUILDER(StrideSliceV2, LAYER_STRIDED_SLICE_V2);

static ShapeTensor clamp(INetworkDefinition* network, const ShapeTensor& x, const ShapeTensor& lowerBound,
        const ShapeTensor& upperBound) {
    return min(network, max(network, x, lowerBound), upperBound);
}

static ShapeTensor bumpIfNegative(INetworkDefinition* network, const ShapeTensor& inputDims,
        const ShapeTensor& indices) {
    const auto signs = clamp(network, indices, shapeVector(-1), shapeVector(0));
    return sub(network, indices, mul(network, signs, inputDims));
}

void decodeOnnxStartsAndEnds(INetworkDefinition* network, const ShapeTensor& inputDims,
        const ShapeTensor& steps, ShapeTensor& starts, ShapeTensor& ends) {
    const auto stepSign = clamp(network, steps, shapeVector(-1), shapeVector(0));
    starts = bumpIfNegative(network, inputDims, starts);
    starts = clamp(network, starts, shapeVector(0), add(network, inputDims, stepSign));
    ends = bumpIfNegative(network, inputDims, ends);
    ends = clamp(network, ends, stepSign, inputDims);
}

ShapeTensor axesToInterlaceSubscripts(const ShapeTensor& axes, int nbDims) {
    std::vector<int> subscripts(nbDims);
    std::iota(subscripts.begin(), subscripts.end(), 0);
    for (int i = 0; i < axes.size(); ++i) {
        subscripts[axes[i]] = nbDims + i;
    }
    return ShapeTensor(1, std::move(subscripts));
}

ShapeTensor computeSliceSizes(INetworkDefinition* network, const ShapeTensor& starts, const ShapeTensor& ends,
        const ShapeTensor& steps, const ShapeTensor& dims) {
    if (steps.isAll(1)) {
        return sub(network, ends, starts);
    } else {
        return sub(network, similar(network, dims, 0), floorDiv(network, sub(network, starts, ends), steps));
    }
}

ILayer* StrideSliceV2TRTLayerBuilder::AddToNetwork(INetworkDefinition* network) {
    StrideSliceV2LayerParam* param = dynamic_cast<StrideSliceV2LayerParam*>(param_);
    auto input_tensors = GetInputITensors();
    auto dims = shapeOf(*input_tensors[0]);

    ShapeTensor begins;
    ShapeTensor ends;
    ShapeTensor axes;
    ShapeTensor strides;

    begins = ShapeTensor(1, std::move(param->begins));
    ends = ShapeTensor(1, std::move(param->ends));
    axes = ShapeTensor(1, std::move(param->axes));
    strides = ShapeTensor(1, std::move(param->strides));

    if (input_tensors.size() >= 2) {
        begins = ShapeTensor(*input_tensors[1], 0);
    }

    if (input_tensors.size() >= 3) {
        ends = ShapeTensor(*input_tensors[2], 0);
    }

    bool isIota = true;
    int j = 0;
    std::vector<int> newAxes;
    newAxes.reserve(axes.size());

    for (int axis : axes) {
        int r = dims.size();
        assert(-r <= axis && axis < r);
        if (axis < 0) {
            axis += r;
        }
        newAxes.push_back(axis);
        isIota &= axis == j;
        ++j;
    }
    axes = ShapeTensor(1, std::move(newAxes));

    assert(std::unordered_set<int>(axes.begin(), axes.end()).size() == static_cast<size_t>(axes.size()));

    if (axes.size() < dims.size() || !isIota) {
        const ShapeTensor subscripts{axesToInterlaceSubscripts(axes, dims.size())};
        begins = interlace(network, similar(network, dims, 0), begins, subscripts);
        ends = interlace(network, dims, ends, subscripts);
        strides = interlace(network, similar(network, dims, 1), strides, subscripts);
    }

    decodeOnnxStartsAndEnds(network, dims, strides, begins, ends);
    const ShapeTensor sizes = computeSliceSizes(network, begins, ends, strides, dims);
    nvinfer1::ISliceLayer* layer = addSlice(network, *input_tensors[0], begins, sizes, strides);
    if (layer != nullptr) {
        layer->setName(layer_name_.c_str());
    }
    return layer;
}

REGISTER_TENSORRT_LAYER_BUILDER(StrideSliceV2, LAYER_STRIDED_SLICE_V2);

}  //  namespace TNN_NS
