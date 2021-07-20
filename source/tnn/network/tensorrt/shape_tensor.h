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


#ifndef TNN_SOURCE_TNN_NETWORK_TENSORRT_SHAPE_TENSOR_H_
#define TNN_SOURCE_TNN_NETWORK_TENSORRT_SHAPE_TENSOR_H_

#include <cassert>
#include <vector>

#include <NvInfer.h>
#ifdef _WIN32
#include <numeric>
#include <functional>
#undef max
#undef min
#endif

#include "tnn/core/macro.h"

namespace TNN_NS {

using namespace nvinfer1;

class ShapeTensor {
public:
    ShapeTensor() = default;

    ShapeTensor(int rank, std::vector<int>&& values);

    explicit ShapeTensor(nvinfer1::ITensor& t, int depth = 0);

    bool rankKnown() const {
        return m_rank != kRANK_UNKNOWN;
    }

    int rank() const {
        assert(rankKnown());
        return m_rank;
    }

    bool sizeKnown() const {
        return m_size != kSIZE_UNKNOWN;
    }

    int size() const {
        assert(sizeKnown());
        return m_size;
    }

    bool allValuesKnown() const {
        return m_all_values_known;
    }

    bool isAll(int value) const;

    using const_iterator = std::vector<int>::const_iterator;

    const_iterator begin() const {
        assert(m_all_values_known);
        return m_values.begin();
    }

    const_iterator end() const {
        assert(m_all_values_known);
        return m_values.end();
    }

    bool valueKnown(int k) const;

    int operator[](int k) const {
        assert(valueKnown(k));
        return m_values[k];
    }

    friend ShapeTensor shapeOf(const ShapeTensor& t);
    friend bool operator==(const ShapeTensor& x, const ShapeTensor& y);
    nvinfer1::ITensor& tensor(INetworkDefinition* network) const;

private:
    mutable int m_depth = -1;

    bool m_all_values_known = false;

    static constexpr int kRANK_UNKNOWN = -1;
    static constexpr int kSIZE_UNKNOWN = -1;

    int m_rank = kRANK_UNKNOWN;

    int m_size = kSIZE_UNKNOWN;

    mutable nvinfer1::ITensor* m_tensor = nullptr;

    std::vector<int> m_values;
};

// Create 1D ShapeTensor of length n filled with value.
// count must be 1D ShapeTensor of size 1.
ShapeTensor fillShapeVector(INetworkDefinition* network, int value, const ShapeTensor& count);

// Create 1D ShapeTensor of length 1 containing given value.
ShapeTensor shapeVector(int value);

// Create 0D ShapeTensor containing the given value.
ShapeTensor shapeScalar(int value);

// Create 1D ShapeTensor containing [0,n).
ShapeTensor iotaShapeVector(int n);

// Create ShapeTensor filled with value that has same shape as exemplar.
// The exemplar must be 1D.
ShapeTensor similar(INetworkDefinition* network, const ShapeTensor& exemplar, int value);

// Elementwise addition
ShapeTensor add(INetworkDefinition* network, const ShapeTensor& x, const ShapeTensor& y);

// Elementwise subtraction
ShapeTensor sub(INetworkDefinition* network, const ShapeTensor& x, const ShapeTensor& y);

// Elementwise multiplication
ShapeTensor mul(INetworkDefinition* network, const ShapeTensor& x, const ShapeTensor& y);

// Elementwise min
ShapeTensor min(INetworkDefinition* network, const ShapeTensor& x, const ShapeTensor& y);

// Elementwise max
ShapeTensor max(INetworkDefinition* network, const ShapeTensor& x, const ShapeTensor& y);

// Elementwise floor division
ShapeTensor floorDiv(INetworkDefinition* network, const ShapeTensor& x, const ShapeTensor& y);

// Elementwise f, for a partial function f defined by:
// f(x,x) = x
// f(1,x) = x
// f(x,1) = x
// Undefined otherwise or if x < 0.
ShapeTensor broadcast(INetworkDefinition* network, const ShapeTensor& x, const ShapeTensor& y);

// Return product of x[i] for i in [first..last), as 0D or one-element 1D tensor of given rank.
ShapeTensor product(INetworkDefinition* network, const ShapeTensor& x, int first, int last, int rank);

// Gather where data is 1D tensor and indices can be 0D or 1D
ShapeTensor gather(INetworkDefinition* network, const ShapeTensor& data, const ShapeTensor& indices);

// Concatenation of two 1D tensors
ShapeTensor concat(INetworkDefinition* network, const ShapeTensor& x, const ShapeTensor& y);

// Return gather(concat(x,y),subscripts)
inline ShapeTensor interlace(INetworkDefinition* network, const ShapeTensor& x, const ShapeTensor& y,
        const ShapeTensor& subscripts) {
    return gather(network, concat(network, x, y), subscripts);
}

// Return shape of a tensor.
ShapeTensor shapeOf(nvinfer1::ITensor& tensor);
ShapeTensor shapeOf(const ShapeTensor& tensor);

// Reshape 0D tensor to 1D tensor.
ShapeTensor convertTo1D(INetworkDefinition* network, const ShapeTensor& tensor);

// Add an ISliceLayer.
nvinfer1::ISliceLayer* addSlice(INetworkDefinition* network, nvinfer1::ITensor& data, const ShapeTensor& starts,
    const ShapeTensor& sizes, const ShapeTensor& strides);

// Add an IShuffleLayer.
// If the result does not need to have its parameters changed, and
// optimizing the no-op case away is okay, use function reshape instead.
//
// In general the default zeroIsPlaceholder=false should be used so
// that reshaping to empty tensors works correctly.  Calling with
// zeroIsPlaceholder=true should happen only when replicating the
// semantics of the ONNX Reshape operator.
nvinfer1::IShuffleLayer* addShuffle(INetworkDefinition* network, nvinfer1::ITensor& data,
    const ShapeTensor& reshapeDims, bool zeroIsPlaceholder = false);

// Add an IFillLayer.
nvinfer1::IFillLayer* addFill(INetworkDefinition* network, const ShapeTensor& shape, nvinfer1::FillOperation op);

// Add a Unsqueeze layer on a given set of axes.
nvinfer1::IShuffleLayer* addUnsqueeze(INetworkDefinition* network,
    nvinfer1::ITensor& tensor, const std::vector<int>& axes);

// Add a Squeeze layer on a given set of axes.
nvinfer1::IShuffleLayer* addSqueeze(INetworkDefinition* network,
    nvinfer1::ITensor& tensor, const std::vector<int>& axes);

// Reshape a tensor.
//
// Treats any zeros in newShape as dimensions, not placeholders.
// Implementation note: does not insert shuffle if it's a no-op.
nvinfer1::ITensor& reshape(INetworkDefinition* network, nvinfer1::ITensor& data, const ShapeTensor& newShape);

}  //  namespace TNN_NS

#endif  //  TNN_SOURCE_TNN_NETWORK_TENSORRT_SHAPE_TENSOR_H_
