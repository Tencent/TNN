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

#include "test/unit_test/layer_test/layer_test.h"
#include "test/unit_test/unit_test_common.h"
#include "test/unit_test/utils/network_helpers.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

    class RoiAlignLayerTest : public LayerTest,
                              public ::testing::WithParamInterface<std::tuple<int, int, int, int, int, int, int, float, DataType>> {
    };

    INSTANTIATE_TEST_SUITE_P(LayerTest, RoiAlignLayerTest,
                             ::testing::Combine(BASIC_BATCH_CHANNEL_SIZE,
                                     // num rois
                                                testing::Values(1, 2),
                                     // output height
                                                testing::Values(3, 4, 5),
                                     // output width
                                                testing::Values(3, 4, 5),
                                     // sampling ratio
                                                testing::Values(0, 2),
                                     // spatial scale
                                                testing::Values(0.125, 0.25, 0.5),
                                     // dtype
                                                testing::Values(DATA_TYPE_FLOAT)));

TEST_P(RoiAlignLayerTest, RoiAlignLayer) {
    // get param
    int batch           = std::get<0>(GetParam());
    int channel         = std::get<1>(GetParam());
    int input_size      = std::get<2>(GetParam());
    int num_rois        = std::get<3>(GetParam());
    int output_height   = std::get<4>(GetParam());
    int output_width    = std::get<5>(GetParam());
    int sampling_ratio  = std::get<6>(GetParam());
    float spatial_scale = std::get<7>(GetParam());
    DataType data_type  = std::get<8>(GetParam());
    DeviceType dev      = ConvertDeviceType(FLAGS_dt);

    if (data_type == DATA_TYPE_INT8 && DEVICE_ARM != dev) {
        GTEST_SKIP();
    }

    // blob desc
    std::vector<BlobDesc> inputs_desc;
    inputs_desc.push_back(CreateInputBlobsDesc(batch, channel, input_size, 1, data_type)[0]);
    inputs_desc.push_back(
            CreateInputBlobsDesc(num_rois, channel, output_height * output_width, 1, data_type)[0]);
    auto outputs_desc = CreateOutputBlobsDesc(1, data_type);

    // param
    RoiAlignLayerParam param;
    param.name = "RoiAlign";
    param.output_height = output_height;
    param.output_width = output_width;
    param.sampling_ratio = sampling_ratio;
    param.spatial_scale = spatial_scale;

    Run(LAYER_ROI_ALIGN, &param, nullptr, inputs_desc, outputs_desc);
}

}  // namespace TNN_NS
