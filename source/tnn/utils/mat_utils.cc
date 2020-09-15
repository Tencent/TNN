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

#include "tnn/utils/dims_vector_utils.h"
#include "tnn/utils/mat_utils.h"
#include "tnn/utils/mat_converter_acc.h"

namespace TNN_NS {

Status MatUtils::Copy(Mat& src, Mat& dst, void* command_queue) {
    DimsVector src_dims = src.GetDims();
    DimsVector dst_dims = dst.GetDims();
    if(DimsVectorUtils::Equal(src_dims, dst_dims) && (src.GetMatType() == dst.GetMatType())) {
        DeviceType device_type = DEVICE_NAIVE;
        // get device type
        DeviceType src_dt = src.GetDeviceType();
        DeviceType dst_dt = dst.GetDeviceType();
        if (src_dt == dst_dt) {
            device_type = src_dt;
        } else if (DEVICE_NAIVE == src_dt || DEVICE_ARM == src_dt) {
            device_type = dst_dt;
        } else if (DEVICE_NAIVE == dst_dt || DEVICE_ARM == dst_dt) {
            device_type = src_dt;
        } else {
            return Status(TNNERR_PARAM_ERR, "src and dst DeviceType need be equal or one is device cpu");
        }
        auto converter = MatConverterManager::Shared()->CreateMatConverterAcc(device_type);
        return converter->Copy(src, dst, command_queue);
    }else {
        return Status(TNNERR_PARAM_ERR, "src and dst dims or MatType not equal"); 
    }
}

Status MatUtils::Resize(Mat& src, Mat& dst, ResizeParam param, void* command_queue) {
    if(param.scale_w == 0) {
         param.scale_w = (double)dst.GetWidth() / src.GetWidth();
    }
    if(param.scale_h == 0) {
         param.scale_h = (double)dst.GetHeight() / src.GetHeight();
    }
    if(src.GetDeviceType() != dst.GetDeviceType() || src.GetMatType() != dst.GetMatType()) {
        return Status(TNNERR_PARAM_ERR, "DeviceType or MatType not equal");
    }
    auto converter = MatConverterManager::Shared()->CreateMatConverterAcc(src.GetDeviceType());
    return converter->Resize(src, dst, param, command_queue);
}

Status MatUtils::Crop(Mat& src, Mat& dst, CropParam param, void* command_queue) {
    if(param.width == 0) {
         param.width = dst.GetWidth();
    }
    if(param.height == 0) {
         param.height = dst.GetHeight();
    }
    if(src.GetDeviceType() != dst.GetDeviceType() || src.GetMatType() != dst.GetMatType()) {
        return Status(TNNERR_PARAM_ERR, "DeviceType or MatType not equal");
    }
    auto converter = MatConverterManager::Shared()->CreateMatConverterAcc(src.GetDeviceType());
    return converter->Crop(src, dst, param, command_queue);
}

Status MatUtils::WarpAffine(Mat& src, Mat& dst, WarpAffineParam param, void* command_queue) {
    if(src.GetDeviceType() != dst.GetDeviceType() || src.GetMatType() != dst.GetMatType()) {
        return Status(TNNERR_PARAM_ERR, "DeviceType or MatType not equal");
    }
    auto converter = MatConverterManager::Shared()->CreateMatConverterAcc(src.GetDeviceType());
    return converter->WarpAffine(src, dst, param, command_queue);
}

Status MatUtils::CvtColor(Mat& src, Mat& dst, ColorConversionType type, void* command_queue) {
    if(src.GetDeviceType() != dst.GetDeviceType()) {
        return Status(TNNERR_PARAM_ERR, "DeviceType or MatType not equal");
    }
    auto converter = MatConverterManager::Shared()->CreateMatConverterAcc(src.GetDeviceType());
    return converter->CvtColor(src, dst, type, command_queue);
}

Status MatUtils::ResizeAndPaste(Mat& src, Mat& dst, ResizeParam param, PasteParam paste_param, void* command_queue) {
    if(src.GetDeviceType() != dst.GetDeviceType() || src.GetMatType() != dst.GetMatType()) {
        return Status(TNNERR_PARAM_ERR, "DeviceType or MatType not equal");
    }
    auto converter = MatConverterManager::Shared()->CreateMatConverterAcc(src.GetDeviceType());
    return converter->ResizeAndPaste(src, dst, param, paste_param, command_queue);
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

    auto converter = MatConverterManager::Shared()->CreateMatConverterAcc(device_type);
    return converter->ConcatMatWithBatch(src_vec, dst, command_queue);
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
