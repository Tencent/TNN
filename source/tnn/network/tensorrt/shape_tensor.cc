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

#include <algorithm>
#include <cassert>
#include <climits>
#include <functional>
#include <numeric>
#include <set>

#include "tnn/network/tensorrt/shape_tensor.h"

namespace TNN_NS {

ShapeTensor::ShapeTensor(int rank, std::vector<int>&& values)
        : m_depth(0)
        , m_all_values_known(true)
        , m_rank(rank)
        , m_size(values.size())
        , m_values(std::move(values)) {
    assert((rank == 0 || rank == 1) && "shape tensor must have rank 0 or 1");
    assert(rank > 0 || m_values.size() == 1);
}

static bool hasAllNonNegativeValues(const std::vector<int>& values) {
    return std::all_of(values.begin(), values.end(), [](int x) { return x >= 0; });
}

ShapeTensor::ShapeTensor(nvinfer1::ITensor& t, int depth)
        : m_depth(depth)
        , m_rank(1)
        , m_tensor(&t) {
    const nvinfer1::Dims dims = t.getDimensions();

    switch (m_depth) {
        case 0 :
            assert(t.getType() == nvinfer1::DataType::kINT32);
            m_rank = dims.nbDims;
            if (m_rank == 0) {
                m_size = 1;
            } else if (m_rank == 1) {
                m_size = dims.d[0];
            } else {
                assert(m_rank == -1);
            }
            break;

        case 1 :
            if (dims.nbDims >= 0) {
                m_size = dims.nbDims;
                m_values.resize(dims.nbDims);
                std::copy_n(dims.d, dims.nbDims, m_values.begin());
                m_all_values_known = hasAllNonNegativeValues(m_values);
            }
            break;

        case 2 :
            m_size = 1;
            if (dims.nbDims >= 0) {
                m_values = {dims.nbDims};
                m_all_values_known = hasAllNonNegativeValues(m_values);
            }
            break;

        case 3 :
            m_depth = 0;
            m_size = 1;
            m_values = {1};
            m_all_values_known = true;
            m_tensor = nullptr;
            break;

        default:
            assert(0);
            break;
    }
}

ShapeTensor shapeVector(int value) {
    return ShapeTensor(1, std::vector<int>({value}));
}

ShapeTensor shapeScalar(int value) {
    return ShapeTensor(0, std::vector<int>({value}));
}

bool ShapeTensor::valueKnown(int k) const {
    assert(0 <= k);
    assert(k < m_size);
    return allValuesKnown() || (m_values.size() == static_cast<size_t>(m_size) && m_values[k] >= 0);
}

bool ShapeTensor::isAll(int x) const {
    assert(m_depth >= 0 && "undefined tensor");
    return allValuesKnown() && std::all_of(begin(), end(), [x](int y) { return x == y; });
}

nvinfer1::ITensor& ShapeTensor::tensor(INetworkDefinition* network) const {
    assert(m_depth >= 0 && "undefined tensor");
    assert(m_depth <= 2);
    if (!m_tensor || m_depth != 0) {
        if (allValuesKnown()) {
            assert(hasAllNonNegativeValues(m_values));
            nvinfer1::Dims dims;
            dims.nbDims = size();
            int count = 1;
            for (int i = 0; i < size(); i++) {
                int value = std::min(INT_MAX-1, m_values[i]);
                dims.d[i] = value;
                count *= value;
            }
            nvinfer1::Weights w{nvinfer1::DataType::kINT32, count == 0 ? nullptr : m_values.data(), count};
            m_tensor = network->addShape(*network->addConstant(dims, w)->getOutput(0))->getOutput(0);
            if (rank() == 0) {
                nvinfer1::IShuffleLayer* shuffle = network->addShuffle(*m_tensor);
                nvinfer1::Dims d{0, {}};
                shuffle->setReshapeDimensions(d);
                m_tensor = shuffle->getOutput(0);
            }
            m_depth = 0;
        } else {
            assert(m_tensor);
            for (; m_depth > 0; --m_depth) {
                m_tensor = network->addShape(*m_tensor)->getOutput(0);
            }
        }
    }
    return *m_tensor;
}

ShapeTensor iotaShapeVector(int n) {
    std::vector<int> values(n);
    std::iota(values.begin(), values.end(), 0);
    return ShapeTensor(1, std::move(values));
}

ShapeTensor similar(INetworkDefinition* network, const ShapeTensor& exemplar, int value) {
    return fillShapeVector(network, value, shapeOf(exemplar));
}

ShapeTensor fillShapeVector(INetworkDefinition* network, int value, const ShapeTensor& count) {
    assert(count.rank() == 1 && "implementation assumes 1D size");
    assert(count.size() == 1 && "implementation assumes 1D size of known size");
    if (count.allValuesKnown()) {
        return ShapeTensor(1, std::vector<int>(count[0], value));
    } else {
        nvinfer1::ISliceLayer* slice = addSlice(network, shapeVector(value).tensor(network), shapeVector(0),
            count, shapeVector(0));
        return ShapeTensor(*slice->getOutput(0));
    }
}

static ShapeTensor op(INetworkDefinition* network, const ShapeTensor& x, const ShapeTensor& y,
        ElementWiseOperation operation, bool commutative, int rightIdentity,
        const std::function<int(int, int)>&& f) {
    assert(!x.rankKnown() || !y.rankKnown() || x.rank() == y.rank());
    if (x.sizeKnown() && y.sizeKnown()) {
        assert(x.size() == 1 || y.size() == 1 || x.size() == y.size());
        if (y.isAll(rightIdentity) && y.size() <= x.size()) {
            return x;
        }
        if (commutative && x.isAll(rightIdentity) && x.size() <= y.size()) {
            return y;
        }
    }
    if (x.allValuesKnown() && y.allValuesKnown()) {
        std::vector<int> values(std::max(x.size(), y.size()));
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = f(x[i % x.size()], y[i % y.size()]);
        }
        return ShapeTensor(x.rank(), std::move(values));
    }
    return ShapeTensor(*network->addElementWise(x.tensor(network), y.tensor(network), operation)->getOutput(0), 0);
}

ShapeTensor add(INetworkDefinition* network, const ShapeTensor& x, const ShapeTensor& y) {
    return op(network, x, y, ElementWiseOperation::kSUM, true, 0, std::plus<int>());
}

ShapeTensor sub(INetworkDefinition* network, const ShapeTensor& x, const ShapeTensor& y) {
    return op(network, x, y, ElementWiseOperation::kSUB, false, 0, std::minus<int>());
}

ShapeTensor mul(INetworkDefinition* network, const ShapeTensor& x, const ShapeTensor& y) {
    return op(network, x, y, ElementWiseOperation::kPROD, true, 1, std::multiplies<int>());
}

ShapeTensor min(INetworkDefinition* network, const ShapeTensor& x, const ShapeTensor& y) {
    return op(network, x, y, ElementWiseOperation::kMIN, true, std::numeric_limits<int>::max(),
        [](int x, int y) { return std::min(x, y); });
}

ShapeTensor max(INetworkDefinition* network, const ShapeTensor& x, const ShapeTensor& y) {
    return op(network, x, y, ElementWiseOperation::kMAX, true, std::numeric_limits<int>::min(),
        [](int x, int y) { return std::max(x, y); });
}

ShapeTensor floorDiv(INetworkDefinition* network, const ShapeTensor& x, const ShapeTensor& y) {
    return op(network, x, y, ElementWiseOperation::kFLOOR_DIV, false, 1, [](int x, int y) {
        assert(y != 0 && "divisor must be non-zero");
        const int d = x / y;
        return d * y == x ? d : d - ((x < 0) ^ (y < 0));
    });
}

ShapeTensor broadcast(INetworkDefinition* network, const ShapeTensor& x, const ShapeTensor& y) {
    return mul(network, max(network, x, y), min(network, x, min(network, y, similar(network, y, 1))));
}

ShapeTensor product(INetworkDefinition* network, const ShapeTensor& x, int first, int last, int rank) {
    assert(first <= last);
    ShapeTensor z(rank, std::vector<int>(1, 1));
    for (int i = first; i < last; ++i) {
        z = mul(network, z, gather(network, x, ShapeTensor(rank, std::vector<int>(1, i))));
    }
    return z;
}

ShapeTensor concat(INetworkDefinition* network, const ShapeTensor& x, const ShapeTensor& y) {
    assert(!x.rankKnown() || x.rank() == 1);
    assert(!y.rankKnown() || y.rank() == 1);
    if (x.sizeKnown() && x.size() == 0) {
        return y;
    }
    if (y.sizeKnown() && y.size() == 0) {
        return x;
    }
    if (x.allValuesKnown() && y.allValuesKnown()) {
        std::vector<int> values(x.size() + y.size());
        auto p = std::copy(x.begin(), x.end(), values.begin());
        std::copy(y.begin(), y.end(), p);
        return ShapeTensor(1, std::move(values));
    }
    nvinfer1::ITensor* const args[2] = {&x.tensor(network), &y.tensor(network)};
    return ShapeTensor(*network->addConcatenation(args, 2)->getOutput(0));
}

ShapeTensor gather(INetworkDefinition* network, const ShapeTensor& data, const ShapeTensor& indices) {
    assert(data.rank() == 1);
    if (indices.allValuesKnown() && std::all_of(indices.begin(), indices.end(),
            [&data](int i) { return data.valueKnown(i); })) {
        std::vector<int> z(indices.size());
        std::transform(indices.begin(), indices.end(), z.begin(), [&data](int i) {
            assert(0 <= i);
            assert(i < data.size());
            return data[i];
        });
        return ShapeTensor(indices.rank(), std::move(z));
    }
    return ShapeTensor(*network->addGather(data.tensor(network), indices.tensor(network), 0)->getOutput(0));
}

ShapeTensor shapeOf(nvinfer1::ITensor& tensor) {
    return ShapeTensor(tensor, 1);
}

ShapeTensor shapeOf(const ShapeTensor& t) {
    assert(t.m_depth >= 0);
    if (t.m_tensor) {
        return ShapeTensor(*t.m_tensor, t.m_depth + 1);
    } else {
        assert(t.rankKnown());
        assert(t.sizeKnown());
        return t.rank() == 0 ? ShapeTensor(0, {}) : ShapeTensor(1, {t.size()});
    }
}

ShapeTensor convertTo1D(INetworkDefinition* network, const ShapeTensor& tensor) {
    assert(tensor.rank() == 0);
    assert(tensor.size() == 1);
    if (tensor.valueKnown(0)) {
        return shapeScalar(tensor[0]);
    }
    return ShapeTensor(*addShuffle(network, tensor.tensor(network), shapeVector(1))->getOutput(0));
}

static nvinfer1::Dims toDims(const ShapeTensor& x) {
    nvinfer1::Dims d{-1, {}};
    if (x.sizeKnown()) {
        d.nbDims = x.size();
        if (x.allValuesKnown()) {
            assert(x.size() <= nvinfer1::Dims::MAX_DIMS);
            std::copy(x.begin(), x.end(), d.d);
        }
    }
    return d;
}

static void setShapeInputIfDynamic(INetworkDefinition* network, nvinfer1::ILayer* layer,
        int inputIndex, const ShapeTensor& x) {
    if (!x.allValuesKnown()) {
        layer->setInput(inputIndex, x.tensor(network));
    }
}

bool operator==(const ShapeTensor& x, const ShapeTensor& y) {
    if (x.allValuesKnown() && y.allValuesKnown()) {
        return x.m_values == y.m_values;
    }
    assert(x.m_tensor || y.m_tensor);
    return x.m_tensor == y.m_tensor && x.m_depth == y.m_depth;
}

nvinfer1::ITensor& reshape(INetworkDefinition* network, nvinfer1::ITensor& data, const ShapeTensor& newShape) {
    const ShapeTensor oldShape = shapeOf(data);
    if (newShape == oldShape) {
        return data;
    }
    return *addShuffle(network, data, newShape)->getOutput(0);
}

nvinfer1::IShuffleLayer* addShuffle(INetworkDefinition* network, nvinfer1::ITensor& data,
        const ShapeTensor& reshapeDims, bool zeroIsPlaceholder) {
    nvinfer1::IShuffleLayer* shuffle = network->addShuffle(data);
    if (reshapeDims.allValuesKnown()) {
        shuffle->setReshapeDimensions(toDims(reshapeDims));
    } else {
        shuffle->setInput(1, reshapeDims.tensor(network));
    }
    shuffle->setZeroIsPlaceholder(zeroIsPlaceholder);
    return shuffle;
}

nvinfer1::ISliceLayer* addSlice(INetworkDefinition* network, nvinfer1::ITensor& data, const ShapeTensor& starts,
        const ShapeTensor& sizes, const ShapeTensor& strides) {
    nvinfer1::ISliceLayer* slice = network->addSlice(data, toDims(starts), toDims(sizes), toDims(strides));
    setShapeInputIfDynamic(network, slice, 1, starts);
    setShapeInputIfDynamic(network, slice, 2, sizes);
    setShapeInputIfDynamic(network, slice, 3, strides);
    return slice;
}

nvinfer1::IFillLayer* addFill(INetworkDefinition* network, const ShapeTensor& shape, nvinfer1::FillOperation op) {
    nvinfer1::IFillLayer* fill = network->addFill(toDims(shape), op);
    setShapeInputIfDynamic(network, fill, 0, shape);
    return fill;
}

nvinfer1::IShuffleLayer* addUnsqueeze(INetworkDefinition* network,
        nvinfer1::ITensor& tensor, const std::vector<int>& axes) {
    const auto dims = shapeOf(tensor);
    const std::set<int> axesSet(axes.begin(), axes.end());

    if (dims.size() + axesSet.size() > nvinfer1::Dims::MAX_DIMS) {
        return nullptr;
    }

    std::vector<int> subscripts(dims.size());
    std::iota(subscripts.begin(), subscripts.end(), 0);
    for (const auto& axis : axesSet) {
        subscripts.insert(subscripts.begin() + axis, dims.size());
    }

    const auto newDims = interlace(network, dims, shapeVector(1), ShapeTensor(1, std::move(subscripts)));
    nvinfer1::IShuffleLayer* unsqueeze_layer = addShuffle(network, tensor, newDims);
    return unsqueeze_layer;
}

nvinfer1::IShuffleLayer* addSqueeze(INetworkDefinition* network, nvinfer1::ITensor& tensor,
        const std::vector<int>& axes) {
    const auto dims = shapeOf(tensor);
    std::vector<int> subscripts(dims.size());
    std::iota(subscripts.begin(), subscripts.end(), 0);
    auto p = std::remove_if(subscripts.begin(), subscripts.end(),
        [axes](int x) { return std::find(axes.begin(), axes.end(), x) != axes.end(); });
    subscripts.resize(p - subscripts.begin());

    auto newDims = gather(network, dims, ShapeTensor(1, std::move(subscripts)));
    nvinfer1::IShuffleLayer* squeeze_layer = addShuffle(network, tensor, newDims);

    return squeeze_layer;
}

}  //  namespace TNN_NS
