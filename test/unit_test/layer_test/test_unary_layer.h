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

#ifndef TNN_TEST_UNIT_TEST_LAYER_TEST_UNARY_LAYER_HPP_
#define TNN_TEST_UNIT_TEST_LAYER_TEST_UNARY_LAYER_HPP_

#include "test/unit_test/layer_test/layer_test.h"
#include "test/unit_test/unit_test_common.h"
#include "test/unit_test/utils/network_helpers.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

class UnaryLayerTest : public LayerTest, public ::testing::WithParamInterface<std::tuple<int, int, int, DataType>> {
public:
    explicit UnaryLayerTest(LayerType type);
    void RunUnaryTest(std::string type_str);

protected:
    LayerType layer_type_;
};

}  // namespace TNN_NS

#endif  // TNN_TEST_UNIT_TEST_LAYER_TEST_UNARY_LAYER_HPP_
