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

#ifndef TNN_TEST_UNIT_TEST_BLOB_CONVERTER_TEST_H_
#define TNN_TEST_UNIT_TEST_BLOB_CONVERTER_TEST_H_

#include <gtest/gtest.h>

#include "test/flags.h"
#include "test/test_utils.h"
#include "tnn/core/abstract_device.h"
#include "tnn/core/common.h"
#include "tnn/core/context.h"
#include "tnn/core/macro.h"
#include "tnn/core/status.h"
#include "tnn/utils/blob_converter.h"
#include "tnn/utils/mat_converter_acc.h"

namespace TNN_NS {

enum class MatConverterType
{
    Copy = 1,
    Resize = 2,
    Crop = 3,
    WarpAffine = 4,
    CvtColor = 5,
    CopyMakeBorder = 6
};

struct MatConverterTestParam
{
    MatConverterType mat_converter_type;
    // Resize
    ResizeParam resize_param;
    // Crop
    CropParam crop_param;
    // WarpAffine
    WarpAffineParam warp_affine_param;
    // CvtColor
    ColorConversionType cvt_type = COLOR_CONVERT_NV12TOBGR;
    // CopyMakeBorder
    CopyMakeBorderParam copy_make_border_param;

    // for Copy
    MatConverterTestParam(MatConverterType converter_type) :
        mat_converter_type(converter_type) { }

    // for Resize
    MatConverterTestParam(MatConverterType converter_type, float scale_w, float scale_h, InterpType type) :
        mat_converter_type(converter_type) {
        resize_param.scale_w    = scale_w;
        resize_param.scale_h    = scale_h;
        resize_param.type       = type;
    }

    // for Crop
    MatConverterTestParam(MatConverterType converter_type, int top_left_x, int top_left_y, int width, int height) :
        mat_converter_type(converter_type) {
        crop_param.top_left_x   = top_left_x;
        crop_param.top_left_y   = top_left_y;
        crop_param.width        = width;
        crop_param.height       = height;
    }

    // for WarpAffine
    MatConverterTestParam(MatConverterType converter_type,
                          float transform00,
                          float transform01,
                          float transform02,
                          float transform10,
                          float transform11,
                          float transform12,
                          InterpType interp_type,
                          BorderType border_type,
                          float border_val) :
        mat_converter_type(converter_type) {
        warp_affine_param.transform[0][0]   = transform00;
        warp_affine_param.transform[0][1]   = transform01;
        warp_affine_param.transform[0][2]   = transform02;
        warp_affine_param.transform[1][0]   = transform10;
        warp_affine_param.transform[1][1]   = transform11;
        warp_affine_param.transform[1][2]   = transform12;
        warp_affine_param.interp_type       = interp_type;
        warp_affine_param.border_type       = border_type;
        warp_affine_param.border_val        = border_val;
    }

    // for CvtColor
    MatConverterTestParam(MatConverterType converter_type, ColorConversionType type) :
        mat_converter_type(converter_type), cvt_type(type) { }

    // for CopyMakeBorder
    MatConverterTestParam(MatConverterType converter_type, int top, int bottom, int left, int right,
                          BorderType border_type, float border_val) :
        mat_converter_type(converter_type) {
        copy_make_border_param.top         = top;
        copy_make_border_param.bottom      = bottom;
        copy_make_border_param.left        = left;
        copy_make_border_param.right       = right;
        copy_make_border_param.border_type = border_type;
        copy_make_border_param.border_val  = border_val;
    }
};

class MatConverterTest : public ::testing::TestWithParam<std::tuple<int, int, int, MatType, MatConverterTestParam>> {
public:
    static void SetUpTestCase();
    static void TearDownTestCase();

protected:
    int Compare(Blob* cpu_blob, Blob* device_blob);
    int CreateTestData(int batch, int channel, int input_size, MatType mat_type, int output_size);
    int DestroyTestData();

    bool OpenCLTestFilter(const DeviceType& device_type, const MatType& mat_type);
    bool MetalTestFilter(const DeviceType& device_type, const MatType& mat_type,
                         const MatConverterType& mat_converter_type, const int batch);
    bool MatChannelCheck(const MatType& mat_type, const int channel);
    bool CvtColorCheck(const DeviceType& device_type, const MatType& mat_type,
                       const MatConverterType& mat_converter_type,
                       const ColorConversionType& cvt_type,
                       const int input_size);
    void GetOutputSize(const MatConverterTestParam& mat_converter_test_param,
                       const MatConverterType& mat_converter_type,
                       const int input_size,
                       int& output_size);

    void* mat_in_data_;
    void* mat_out_ref_data_;
    void* mat_out_dev_data_;

    int out_size_;

    static AbstractDevice* cpu_;
    static AbstractDevice* device_;
    static Context* cpu_context_;
    static Context* device_context_;
};

}  // namespace TNN_NS

#endif  // TNN_TEST_UNIT_TEST_BLOB_CONVERTER_TEST_H_
