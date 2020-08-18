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

#include "tnn/utils/mat_utils.h"
#include "tnn/utils/mat_converter.h"

namespace TNN_NS {

Status MatUtils::Resize(Mat& src, Mat& dst, ResizeParam param, void* command_queue) {
    MatConverter convert(&src, &dst);
    return convert.Resize(src, dst, param, command_queue);
}

Status MatUtils::ResizeAndPaste(Mat& src, Mat& dst, ResizeParam param, PasteParam paste_param, void* command_queue) {
    MatConverter convert(&src, &dst);
    return convert.ResizeAndPaste(src, dst, param, paste_param, command_queue);
}

Status MatUtils::Crop(Mat& src, Mat& dst, CropParam param, void* command_queue) {
    if (dst.GetHeight() != param.height || dst.GetWidth() != param.width) {
        return Status(TNNERR_PARAM_ERR, "crop size not match with dst mat");
    }

    MatConverter convert(&src, &dst);
    return convert.Crop(src, dst, param, command_queue);
}

Status MatUtils::WarpAffine(Mat& src, Mat& dst, WarpAffineParam param, void* command_queue) {
    MatConverter convert(&src, &dst);
    return convert.WarpAffine(src, dst, param, command_queue);
}

Status MatUtils::ConcatMatWithBatch(std::vector<Mat>& src_vec, Mat& dst, void* command_queue) {
    if(src_vec.size() <= 0) {
        return Status(TNNERR_PARAM_ERR, "input mat vector size is 0");
    }

    DeviceType device_type = src_vec[0].GetDeviceType();
    MatType mat_type = src_vec[0].GetMatType();
    int channel = src_vec[0].GetChannel();
    int height = src_vec[0].GetHeight();
    int width = src_vec[0].GetWidth();
    for (auto elem : src_vec) {
        if (device_type != elem.GetDeviceType()) {
            return Status(TNNERR_PARAM_ERR, "the device type of input mat vector is not same");
        }

        if (mat_type != elem.GetMatType()) {
            return Status(TNNERR_PARAM_ERR, "the mat type of input mat vector is not same");
        }

        if (channel != elem.GetChannel()) {
            return Status(TNNERR_PARAM_ERR, "the channel of input mat vector is not same");
        }

        if (height != elem.GetHeight()) {
            return Status(TNNERR_PARAM_ERR, "the height of input mat vector is not same");
        }

        if (width != elem.GetWidth()) {
            return Status(TNNERR_PARAM_ERR, "the width of input mat vector is not same");
        }
    }

    MatConverter convert(&src_vec[0], &dst);
    return convert.ConcatMatWithBatch(src_vec, dst, command_queue);
}

Status MatUtils::GetMatByteSize(Mat& src, int& byte_size) {
    int N = src.GetBatch();
    int C = src.GetChannel();
    int H = src.GetHeight();
    int W = src.GetWidth();

    MatType mat_type = src.GetMatType();

    if (NCHW_FLOAT == mat_type) {
        byte_size = N * C * W * H * sizeof(float);
    } else if (N8UC3 == mat_type) {
        byte_size = N * 3 * W * H;
    } else if (N8UC4 == mat_type) {
        byte_size = N * 4 * W * H;
    } else if (NGRAY == mat_type) {
        byte_size = N * 1 * W * H;
    } else if (NNV12 == mat_type || NNV21 == mat_type) {
        if (H % 2 != 0 || W %2 != 0) {
            LOGE("invaild width or height for YUV (need to be even): %d x %d\n", H, W);
            return Status(TNNERR_PARAM_ERR, "invaild width or height for YUV");
        }
        byte_size = N * 3 * W * H / 2;
    } else {
        LOGE("not support this mat type: %d\n", mat_type);
        return Status(TNNERR_PARAM_ERR, "not support this mat type");
    }

    return TNN_OK;
}


}  // namespace TNN_NS
