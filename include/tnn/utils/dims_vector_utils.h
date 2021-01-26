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

#ifndef TNN_INCLUDE_TNN_UTILS_DIMS_VECTOR_UTILS_H_
#define TNN_INCLUDE_TNN_UTILS_DIMS_VECTOR_UTILS_H_

#include <algorithm>

#include "tnn/core/common.h"
#include "tnn/core/macro.h"
#include "tnn/core/status.h"

namespace TNN_NS {

class PUBLIC DimsVectorUtils {
public:
    // @brief all dims product, [start_index, end_index)
    // @param dims
    static int Count(DimsVector dims, int start_index = 0, int end_index = -1);

    // @brief max of dims0 and dims1, [start_index, end_index)
    static DimsVector Max(DimsVector dims0, DimsVector dims1, int start_index = 0, int end_index = -1);

    // @brief equal of dims0 and dims1, [start_index, end_index)
    static bool Equal(DimsVector dims0, DimsVector dims1, int start_index = 0, int end_index = -1);
    
    // @brief like onnx expand. The broadcast rule is similar to numpy.array(input) * numpy.ones(shape): Dimensions are right alignment.
    // Example:
    // shape = [3, 4]
    // data = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape)
    // print(data)
    //    [[ 1.  2.  3.  4.]
    //     [ 5.  6.  7.  8.]
    //     [ 9. 10. 11. 12.]]
    // new_shape = [2,1,1, 4]
    // expanded = data * np.ones(new_shape, dtype=np.float32)
    // print("数组形状：",expanded.shape)
    // 数组形状： (2, 1, 3, 4)
    // print(expanded)
    //    [[[[ 1.  2.  3.  4.]
    //       [ 5.  6.  7.  8.]
    //       [ 9. 10. 11. 12.]]]
    //
    //
    //     [[[ 1.  2.  3.  4.]
    //       [ 5.  6.  7.  8.]
    //       [ 9. 10. 11. 12.]]]]
    static DimsVector Expand(DimsVector dims0, DimsVector dims1, Status *status);
    
    // @brief reshape op to reshape input dims
    static DimsVector Reshape(const DimsVector input_dims, const DimsVector shape,
                              const int axis, const int num_axes, Status *status);
    
    // @brief strideslice op to slice input dims, it also rectify begins and ends in case value < 0 or = INT_MAX
    static DimsVector StrideSlice(const DimsVector input_dims,
                                  DimsVector& begins, DimsVector& ends, const DimsVector strides,
                                  const DimsVector axes, Status *status);
    
    // @brief upsample/resize op to resize input dims
    static DimsVector Upsample(const DimsVector input_dims,
                                  std::vector<float> scales, std::vector<int> sizes, int mode, Status *status);
    // @brief PadV2 to calc input dims index
    static DimsVector Pad(const DimsVector output_index, DimsVector input_dims, DimsVector pads,
                          int type, Status *status);
    
    // @brief range op to infer output dims
    static DimsVector Range(const RangeData start, const RangeData limit,
                            const RangeData delta, DataType type, Status *status);

    // @brief NCHW dims vector to NHWC dims vector
    static DimsVector NCHW2NHWC(DimsVector dims);

    // @brief NHWC dims vector to NCHW
    static DimsVector NHWC2NCHW(DimsVector dims);
    
    static bool IsInBox(const DimsVector index, const DimsVector box);
    
    // @brief Increase index by offset, bounded by shape
    // @param index
    static DimsVector IncreaseIndex(DimsVector index, const DimsVector shape, int offset = 1);
    
    // @brief compute stride of shape index by offset, bounded by shape
    // @param shape
    static DimsVector StrideOfShape(DimsVector shape);
};

}  // namespace TNN_NS

#endif  // TNN_INCLUDE_TNN_UTILS_DIMS_VECTOR_UTILS_H_
