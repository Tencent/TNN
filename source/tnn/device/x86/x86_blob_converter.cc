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

#include "tnn/core/macro.h"
#include "tnn/core/blob_int8.h"
#include "tnn/device/x86/x86_blob_converter.h"
#include "tnn/device/x86/x86_mat_util.h"
#include "tnn/utils/data_format_converter.h"
#include "tnn/utils/naive_compute.h"
#include "tnn/utils/string_utils_inner.h"

namespace TNN_NS {
using namespace x86;

std::string X86BlobConverterAcc::GetUniqueBlobConvertKey(MatType mat_type, DataType data_type,
                                                         BlobConvertDirection cvt_dir) {
    return ToString(mat_type) + "_" + ToString(data_type) + "_" + ToString(cvt_dir);
}

std::map<std::string, X86BlobConvertFunc>& X86BlobConverterAcc::GetBlobConvertFuncMap() {
    static std::map<std::string, X86BlobConvertFunc> cvt_map;
    return cvt_map;
}

Status X86BlobConverterAcc::RegisterBlobConvertFunc(MatType mat_type, DataType data_type,
                                                    BlobConvertDirection cvt_dir, X86BlobConvertFunc cvt_func) {
    auto& cvt_map       = GetBlobConvertFuncMap();
    const auto& cvt_key = GetUniqueBlobConvertKey(mat_type, data_type, cvt_dir);
    cvt_map[cvt_key] = cvt_func;
    return TNN_OK;
}

Status X86BlobConverterAcc::GetBlobConvertFunc(MatType mat_type, DataType data_type,
                                               BlobConvertDirection cvt_dir, X86BlobConvertFunc& cvt_func) {
    const auto& cvt_map = GetBlobConvertFuncMap();
    const auto& cvt_key = GetUniqueBlobConvertKey(mat_type, data_type, cvt_dir);
    if (cvt_map.find(cvt_key) == cvt_map.end() || cvt_map.at(cvt_key) == nullptr) {
        LOGE("X86BlobConverterAcc::GetBlobConvertFunc, convert type not support yet. mat_type:%d data_type:%d cvt_dir:%d\n", mat_type, data_type, cvt_dir);
        return Status(TNNERR_PARAM_ERR, "X86BlobConverterAcc::GetBlobConvertFunc, convert type not support yet");
    }
    cvt_func = cvt_map.at(cvt_key);
    return TNN_OK;
}

Status X86BlobConverterAcc::ConvertToMatAsync(Mat &image, MatConvertParam param, void *command_queue) {
    Status ret = TNN_OK;
    if (blob_ == nullptr) {
        return Status(TNNERR_NULL_PARAM, "input/output blob is null");
    }

    auto desc       = blob_->GetBlobDesc();
    if (desc.data_type == DATA_TYPE_INT8) {
        auto dims       = desc.dims;
        auto hw         = DimsVectorUtils::Count(dims, 2);
        auto c          = DimsFunctionUtils::GetDim(dims, 1);
        auto c_r4       = ROUND_UP(c, 4);

        if (fused_int8_scale.size() < c_r4) {
            fused_int8_scale.resize(c_r4);
            fused_int8_bias.resize(c_r4);
        }
        auto scale_handle = reinterpret_cast<BlobInt8 *>(blob_)->GetIntResource()->scale_handle;
        auto scale_data   = scale_handle.force_to<float *>();
        auto scale_count  = scale_handle.GetDataCount();
        for (int i = 0; i < dims[1]; i++) {
            auto scale_idx      = scale_count == 1 ? 0 : i;
            fused_int8_scale[i] = param.scale[i] * scale_data[scale_idx];
            fused_int8_bias[i]  = param.bias[i];
        }

        auto cvt_handle_ptr = handle_ptr<char *>(blob_->GetHandle());

        ret = GetBlobConvertFunc(image.GetMatType(), DATA_TYPE_INT8, CVT_DIR_BLOB2MAT, cvt_func_);
        if (ret == TNN_OK) {
            return cvt_func_(image, cvt_handle_ptr, param, dims, hw, c_r4, fused_int8_scale, fused_int8_bias);
        } else {
            return ret;
        }
    } else {
        return DefaultBlobConverterAcc::ConvertToMatAsync(image, param, command_queue);
    }
}

Status X86BlobConverterAcc::ConvertFromMatAsync(Mat &image, MatConvertParam param, void *command_queue) {
    Status ret = TNN_OK;
    if (blob_ == nullptr) {
        return Status(TNNERR_NULL_PARAM, "input/output blob_ is null");
    }
    auto desc       = blob_->GetBlobDesc();
    if (desc.data_type == DATA_TYPE_INT8) {
        auto dims       = desc.dims;
        auto hw         = DimsVectorUtils::Count(dims, 2);
        auto c          = DimsFunctionUtils::GetDim(dims, 1);
        auto c_r4       = ROUND_UP(c, 4);

        if (fused_int8_scale.size() < c_r4) {
            fused_int8_scale.resize(c_r4);
            fused_int8_bias.resize(c_r4);
        }
        auto scale_handle = reinterpret_cast<BlobInt8 *>(blob_)->GetIntResource()->scale_handle;
        auto scale_data   = scale_handle.force_to<float *>();
        auto scale_count  = scale_handle.GetDataCount();
        for (int i = 0; i < dims[1]; i++) {
            auto scale_idx = scale_count == 1 ? 0 : i;
            if (scale_data[scale_idx] != 0) {
                fused_int8_scale[i] = param.scale[i] / scale_data[scale_idx];
                fused_int8_bias[i]  = param.bias[i] / scale_data[scale_idx];
            } else {
                fused_int8_scale[i] = 0;
                fused_int8_bias[i]  = 0;
            }
        }

        auto cvt_handle_ptr = handle_ptr<char *>(blob_->GetHandle());

        ret = GetBlobConvertFunc(image.GetMatType(), DATA_TYPE_INT8, CVT_DIR_MAT2BLOB, cvt_func_);
        if (ret == TNN_OK) {
            ret = cvt_func_(image, cvt_handle_ptr, param, dims, hw, c_r4, fused_int8_scale, fused_int8_bias);
        } else {
            return ret;
        }
    } else {
        return DefaultBlobConverterAcc::ConvertFromMatAsync(image, param, command_queue);
    }

    return ret;
}

Status X86BlobConverterAcc::ConvertToMat(Mat &image, MatConvertParam param, void *command_queue) {
    return ConvertToMatAsync(image, param, command_queue);
}

Status X86BlobConverterAcc::ConvertFromMat(Mat &image, MatConvertParam param, void *command_queue) {
    return ConvertFromMatAsync(image, param, command_queue);
}

DECLARE_BLOB_CONVERTER_CREATER(X86);
REGISTER_BLOB_CONVERTER(X86, DEVICE_X86);

/*
Convert From Mat and Convert To Mat Implementions
*/

/*
convert data type from uint8 to int8, data format from nhw4 2 nhw4
*/
template <bool reverse_channel>
static void BGRAToBlobImpl(const uint8_t *src, int8_t *dst, const float *scale, const float *bias,
                           int hw, int channel) {
    for (int i = 0; i < hw; ++i) {
        dst[4 * i + 0] = float2int8(scale[0] * src[4 * i + (reverse_channel ? 2 : 0)] + bias[0]);
        dst[4 * i + 1] = float2int8(scale[1] * src[4 * i + 1] + bias[1]);
        dst[4 * i + 2] = float2int8(scale[2] * src[4 * i + (reverse_channel ? 0 : 2)] + bias[2]);
        dst[4 * i + 3] = float2int8(scale[3] * src[4 * i + 3] + bias[3]);
        if (channel == 3) {
            dst[4 * i + 3] = 0;
        }
    }
}

/*
convert data type from uint8 to int8, data format from nhw3 2 nhw4
*/
template <bool reverse_channel>
static void BGRToBlobImpl(const uint8_t *src, int8_t *dst, const float *scale, const float *bias, int hw) {
    for (int i = 0; i < hw; ++i) {
        dst[4 * i + 0] = float2int8(scale[0] * src[3 * i + (reverse_channel ? 2 : 0)] + bias[0]);
        dst[4 * i + 1] = float2int8(scale[1] * src[3 * i + 1] + bias[1]);
        dst[4 * i + 2] = float2int8(scale[2] * src[3 * i + (reverse_channel ? 0 : 2)] + bias[2]);
        dst[4 * i + 3] = 0;
    }
}

static void BGRAToBlob(const uint8_t *src, int8_t *dst, const float *scale, const float *bias, int hw,
                       bool reverse_channel, int channel) {
    if (reverse_channel) {
        BGRAToBlobImpl<true>(src, dst, scale, bias, hw, channel);
    } else {
        BGRAToBlobImpl<false>(src, dst, scale, bias, hw, channel);
    }
}

static void BGRToBlob(const uint8_t *src, int8_t *dst, const float *scale, const float *bias, int hw,
                       bool reverse_channel) {
    if (reverse_channel) {
        BGRToBlobImpl<true>(src, dst, scale, bias, hw);
    } else {
        BGRToBlobImpl<false>(src, dst, scale, bias, hw);
    }
}

/*
convert data type from uint8 to int8, data format from nhw1 2 nhwc
*/
static void GrayToBlob(const uint8_t *src, int8_t *dst, const float scale, const float bias, int hw) {
    int i = 0;
    memset(dst, 0, hw * 4 * sizeof(int8_t));
    for (; i < hw; ++i) {
        dst[4 * i] = float2int8(scale * src[i] + bias);
    }
}

static Mat GetBGRFromYUV(Mat& image, const DimsVector& dims, const int hw, bool is_nv12) {
    Mat bgr(DEVICE_X86, N8UC3, image.GetDims());
    for (int n = 0; n < dims[0]; n++) {
        if (is_nv12) {
            NV12ToBGR(reinterpret_cast<uint8_t *>(image.GetData()) + n * 3 * hw / 2,
                      reinterpret_cast<uint8_t *>(bgr.GetData()) + n * 3 * hw, dims[2], dims[3]);
        } else {
            NV21ToBGR(reinterpret_cast<uint8_t *>(image.GetData()) + n * 3 * hw / 2,
                      reinterpret_cast<uint8_t *>(bgr.GetData()) + n * 3 * hw, dims[2], dims[3]);
        }
    }
    return bgr;
}

static void NCHWToBlob(const float *src, int8_t *dst, int channel, int hw, float *scale) {
    int idx  = 0;
    int c_r4 = ROUND_UP(channel, 4);
    memset(dst, 0, hw * c_r4);
    for (int c = 0; c < channel; ++c) {
        int8_t *dst_c = dst + c;
        for (int cur_hw = 0; cur_hw < hw; ++cur_hw) {
            dst_c[cur_hw * c_r4] = float2int8(src[idx++] * scale[c]);
        }
    }
}

static Status ConvertN8UC4ToInt8Blob(Mat& image, char* handle_ptr,
                                     const MatConvertParam& param, const DimsVector& dims,
                                     const int hw, const int c_r4,
                                     std::vector<float>& fused_int8_scale, std::vector<float>& fused_int8_bias) {
    for (int n = 0; n < dims[0]; n++) {
        BGRAToBlob(reinterpret_cast<uint8_t *>(image.GetData()) + n * 4 * hw,
                   reinterpret_cast<int8_t *>(handle_ptr) + n * 4 * hw,
                   fused_int8_scale.data(), fused_int8_bias.data(), hw, param.reverse_channel, dims[1]);
    }
    return TNN_OK;
}

static Status ConvertN8UC3ToInt8Blob(Mat& image, char* handle_ptr,
                                     const MatConvertParam& param, const DimsVector& dims,
                                     const int hw, const int c_r4,
                                     std::vector<float>& fused_int8_scale, std::vector<float>& fused_int8_bias) {
    for (int n = 0; n < dims[0]; n++) {
        BGRToBlob(reinterpret_cast<uint8_t *>(image.GetData()) + n * 3 * hw,
                  reinterpret_cast<int8_t *>(handle_ptr) + n * 4 * hw,
                  fused_int8_scale.data(), fused_int8_bias.data(), hw, param.reverse_channel);
    }
    return TNN_OK;
}

static Status ConvertNGRAYToInt8Blob(Mat& image, char* handle_ptr,
                                     const MatConvertParam& param, const DimsVector& dims,
                                     const int hw, const int c_r4,
                                     std::vector<float>& fused_int8_scale, std::vector<float>& fused_int8_bias) {
    for (int n = 0; n < dims[0]; n++) {
        GrayToBlob(reinterpret_cast<uint8_t *>(image.GetData()) + n * 1 * hw,
                   reinterpret_cast<int8_t *>(handle_ptr) + n * 4 * hw,
                   fused_int8_scale[0], fused_int8_bias[0], hw);
    }
    return TNN_OK;
}

static Status ConvertNNV12ToInt8Blob(Mat& image, char* handle_ptr,
                                     const MatConvertParam& param, const DimsVector& dims,
                                     const int hw, const int c_r4,
                                     std::vector<float>& fused_int8_scale, std::vector<float>& fused_int8_bias) {
    Mat bgr = GetBGRFromYUV(image, dims, hw, true);
    return ConvertN8UC3ToInt8Blob(bgr, handle_ptr, param, dims, hw, c_r4, fused_int8_scale, fused_int8_bias);
}

static Status ConvertNNV21ToInt8Blob(Mat& image, char* handle_ptr,
                                     const MatConvertParam& param, const DimsVector& dims,
                                     const int hw, const int c_r4,
                                     std::vector<float>& fused_int8_scale, std::vector<float>& fused_int8_bias) {
    Mat bgr = GetBGRFromYUV(image, dims, hw, false);
    return ConvertN8UC3ToInt8Blob(bgr, handle_ptr, param, dims, hw, c_r4, fused_int8_scale, fused_int8_bias);
}

static Status ConvertNCHWFloatToInt8Blob(Mat& image, char* handle_ptr,
                                         const MatConvertParam& param, const DimsVector& dims,
                                         const int hw, const int c_r4,
                                         std::vector<float>& fused_int8_scale, std::vector<float>& fused_int8_bias) {
    for (int n = 0; n < dims[0]; n++) {
        NCHWToBlob(reinterpret_cast<float *>(image.GetData()) + n * dims[1] * hw,
                   reinterpret_cast<int8_t *>(handle_ptr) + n * c_r4 * hw, dims[1], hw,
                   fused_int8_scale.data());
    }
    return TNN_OK;
}

static Status ConvertInt8MatToInt8Blob(Mat& image, char* handle_ptr, const MatConvertParam& param,
                                       const DimsVector& dims, const int hw, const int c_r4,
                                       std::vector<float>& fused_int8_scale, std::vector<float>& fused_int8_bias) {
    auto batch   = DimsFunctionUtils::GetDim(dims, 0);
    auto channel = DimsFunctionUtils::GetDim(dims, 1);
    return DataFormatConverter::ConvertFromNCHWToNHWC4Int8(reinterpret_cast<int8_t*>(image.GetData()),
                                                           reinterpret_cast<int8_t*>(handle_ptr), batch, channel, hw);
}

REGISTER_X86_BLOB_CONVERT_FUNC(N8UC4,               DATA_TYPE_INT8,  CVT_DIR_MAT2BLOB, ConvertN8UC4ToInt8Blob)
REGISTER_X86_BLOB_CONVERT_FUNC(N8UC3,               DATA_TYPE_INT8,  CVT_DIR_MAT2BLOB, ConvertN8UC3ToInt8Blob)
REGISTER_X86_BLOB_CONVERT_FUNC(NGRAY,               DATA_TYPE_INT8,  CVT_DIR_MAT2BLOB, ConvertNGRAYToInt8Blob)
REGISTER_X86_BLOB_CONVERT_FUNC(NNV12,               DATA_TYPE_INT8,  CVT_DIR_MAT2BLOB, ConvertNNV12ToInt8Blob)
REGISTER_X86_BLOB_CONVERT_FUNC(NNV21,               DATA_TYPE_INT8,  CVT_DIR_MAT2BLOB, ConvertNNV21ToInt8Blob)
REGISTER_X86_BLOB_CONVERT_FUNC(NCHW_FLOAT,          DATA_TYPE_INT8,  CVT_DIR_MAT2BLOB, ConvertNCHWFloatToInt8Blob)
REGISTER_X86_BLOB_CONVERT_FUNC(RESERVED_INT8_TEST,  DATA_TYPE_INT8,  CVT_DIR_MAT2BLOB, ConvertInt8MatToInt8Blob)

template <bool reverse_channel>
static void BlobToBGRAImpl(const int8_t *src, uint8_t *dst, const float *scale, const float *bias,
                           int hw, int channel) {
    for (int i = 0; i < hw; ++i) {
        dst[4 * i + 0] = float2uint8(reverse_channel ? (scale[2] * src[4 * i + 2] + bias[2]) :
                                                       (scale[0] * src[4 * i + 0] + bias[0]));
        dst[4 * i + 1] = float2uint8(scale[1] * src[4 * i + 1] + bias[1]);
        dst[4 * i + 2] = float2uint8(reverse_channel ? (scale[0] * src[4 * i + 0] + bias[0]) :
                                                       (scale[2] * src[4 * i + 2] + bias[2]));
        if (channel == 4) {
            dst[4 * i + 3] = float2uint8(scale[3] * src[4 * i + 3] + bias[3]);
        }
    }
}

template <bool reverse_channel>
static void BlobToBGRImpl(const int8_t *src, uint8_t *dst, const float *scale, const float *bias, int hw) {
    for (int i = 0; i < hw; ++i) {
        dst[3 * i + 0] = float2uint8(reverse_channel ? (scale[2] * src[4 * i + 2] + bias[2]) :
                                                       (scale[0] * src[4 * i + 0] + bias[0]));
        dst[3 * i + 1] = float2uint8(scale[1] * src[4 * i + 1] + bias[1]);
        dst[3 * i + 2] = float2uint8(reverse_channel ? (scale[0] * src[4 * i + 0] + bias[0]) :
                                                       (scale[2] * src[4 * i + 2] + bias[2]));
    }
}

static void BlobToBGRA(const int8_t *src, uint8_t *dst, const float *scale, const float *bias, int hw,
                       bool reverse_channel, int channel) {
    if (reverse_channel) {
        BlobToBGRAImpl<true>(src, dst, scale, bias, hw, channel);
    } else {
        BlobToBGRAImpl<false>(src, dst, scale, bias, hw, channel);
    }
}

static void BlobToBGR(const int8_t *src, uint8_t *dst, const float *scale, const float *bias, int hw,
                      bool reverse_channel) {
    if (reverse_channel) {
        BlobToBGRImpl<true>(src, dst, scale, bias, hw);
    } else {
        BlobToBGRImpl<false>(src, dst, scale, bias, hw);
    }
}

static void Int8BlobToNCHW(const int8_t *src, float *dst, int channel, int hw, float *scale, float *bias) {
    int cur_hw;
    int c;
    int idx  = 0;
    int c_r4 = ROUND_UP(channel, 4);
    for (c = 0; c < channel; ++c) {
        auto *src_c = src + c;
        for (cur_hw = 0; cur_hw < hw; ++cur_hw) {
            dst[idx++] = src_c[c_r4 * cur_hw] * scale[c] + bias[c];
        }
    }
}

static Status ConvertInt8BlobToN8UC4(Mat& image, char* handle_ptr,
                                     const MatConvertParam& param, const DimsVector& dims,
                                     const int hw, const int c_r4,
                                     std::vector<float>& fused_int8_scale, std::vector<float>& fused_int8_bias) {
    for (int n = 0; n < dims[0]; n++) {
        BlobToBGRA(reinterpret_cast<int8_t *>(handle_ptr) + n * 4 * hw,
                   reinterpret_cast<uint8_t *>(image.GetData()) + n * 4 * hw,
                   fused_int8_scale.data(), fused_int8_bias.data(), hw, param.reverse_channel, dims[1]);
    }
    return TNN_OK;
}

static Status ConvertInt8BlobToN8UC3(Mat& image, char* handle_ptr,
                                     const MatConvertParam& param, const DimsVector& dims,
                                     const int hw, const int c_r4,
                                     std::vector<float>& fused_int8_scale, std::vector<float>& fused_int8_bias) {
    for (int n = 0; n < dims[0]; n++) {
        BlobToBGR(reinterpret_cast<int8_t *>(handle_ptr) + n * 4 * hw,
                   reinterpret_cast<uint8_t *>(image.GetData()) + n * 3 * hw,
                   fused_int8_scale.data(), fused_int8_bias.data(), hw, param.reverse_channel);
    }
    return TNN_OK;
}

static Status ConvertInt8BlobToNCHWFloat(Mat& image, char* handle_ptr,
                                         const MatConvertParam& param, const DimsVector& dims,
                                         const int hw, const int c_r4,
                                         std::vector<float>& fused_int8_scale, std::vector<float>& fused_int8_bias) {
    for (int n = 0; n < dims[0]; n++) {
        Int8BlobToNCHW(reinterpret_cast<int8_t *>(handle_ptr) + n * c_r4 * hw,
                       reinterpret_cast<float *>(image.GetData()) + n * dims[1] * hw, dims[1], hw,
                       fused_int8_scale.data(), fused_int8_bias.data());
    }
    return TNN_OK;
}

static Status ConvertInt8BlobToInt8Mat(Mat& image, char* handle_ptr, const MatConvertParam& param,
                                       const DimsVector& dims, const int hw, const int c_r4,
                                       std::vector<float>& fused_int8_scale, std::vector<float>& fused_int8_bias) {
    auto batch   = DimsFunctionUtils::GetDim(dims, 0);
    auto channel = DimsFunctionUtils::GetDim(dims, 1);
    return DataFormatConverter::ConvertFromNHWC4ToNCHWInt8(
        reinterpret_cast<int8_t*>(handle_ptr), reinterpret_cast<int8_t*>(image.GetData()), batch, channel, hw);
}

REGISTER_X86_BLOB_CONVERT_FUNC(N8UC4,               DATA_TYPE_INT8,  CVT_DIR_BLOB2MAT, ConvertInt8BlobToN8UC4)
REGISTER_X86_BLOB_CONVERT_FUNC(N8UC3,               DATA_TYPE_INT8,  CVT_DIR_BLOB2MAT, ConvertInt8BlobToN8UC3)
REGISTER_X86_BLOB_CONVERT_FUNC(NCHW_FLOAT,          DATA_TYPE_INT8,  CVT_DIR_BLOB2MAT, ConvertInt8BlobToNCHWFloat)
REGISTER_X86_BLOB_CONVERT_FUNC(RESERVED_INT8_TEST,  DATA_TYPE_INT8,  CVT_DIR_BLOB2MAT, ConvertInt8BlobToInt8Mat)

}  // namespace TNN_NS
