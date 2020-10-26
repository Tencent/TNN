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

#include "tnn/utils/blob_converter_default.h"

#include <algorithm>
#include <cstring>

#include "tnn/core/blob_int8.h"
#include "tnn/core/macro.h"
#include "tnn/utils/naive_compute.h"
#include "tnn/utils/bfp16.h"
#include "tnn/utils/bfp16_utils.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

DefaultBlobConverterAcc::DefaultBlobConverterAcc(Blob *blob) : BlobConverterAcc(blob) {}
DefaultBlobConverterAcc::~DefaultBlobConverterAcc() {}

static uint8_t saturate_cast(float data) {
    data += 0.5;
    data = std::min(std::max(data, 0.0f), 255.0f);
    return static_cast<uint8_t>(data);
}

/*
 * Convert an uint8 BGR / BGRA image to nchw float blob
 */
static void BGRAToBlob(const uint8_t *src, float *dst, float *scale, float *bias, int channel, int hw) {
    auto dst_c0 = dst, dst_c1 = dst + hw;
    auto dst_c2 = dst + hw * 2, dst_c3 = dst + hw * 3;
    for (int i = 0; i < hw; ++i) {
        dst_c0[i] = scale[0] * src[4 * i + 0] + bias[0];
        dst_c1[i] = scale[1] * src[4 * i + 1] + bias[1];
        dst_c2[i] = scale[2] * src[4 * i + 2] + bias[2];
        if (channel == 4)
            dst_c3[i] = scale[3] * src[4 * i + 3] + bias[3];
    }
}

/*
 * Convert an uint8 single channel image to nchw float blob
 */
static void GrayToBlob(const uint8_t *src, float *dst, float scale, float bias, int hw) {
    for (int i = 0; i < hw; ++i) {
        dst[i] = scale * src[i] + bias;
    }
}

/*
 * Convert an uint8 BGR image to nchw float blob
 */
static void BGRToBlob(const uint8_t *src, float *dst, float *scale, float *bias, int hw) {
    auto dst_c0 = dst, dst_c1 = dst + hw, dst_c2 = dst + hw * 2;
    for (int i = 0; i < hw; ++i) {
        dst_c0[i] = scale[0] * src[3 * i + 0] + bias[0];
        dst_c1[i] = scale[1] * src[3 * i + 1] + bias[1];
        dst_c2[i] = scale[2] * src[3 * i + 2] + bias[2];
    }
}

/*
 * Convert a nchw float mat to/from nchw float blob
 */
static void NCHWConvert(const float *src, float *dst, float *scale, float *bias, int channel, int hw) {
    for (int c = 0; c < channel; ++c) {
        for (int i = 0; i < hw; ++i) {
            int data_pos = c * hw + i;
            dst[data_pos] = scale[c] * src[data_pos] + bias[c];
        }
    }
}

/*
 * Convert a nchw float blob to BGRA
 * input blob must have 3 or 4 channels
 */
static void BlobToBGRA(const float *src, uint8_t *dst, float *scale, float *bias, int channel, int hw) {
    auto src_c0 = src, src_c1 = src + hw;
    auto src_c2 = src + hw * 2, src_c3 = src + hw * 3;
    for (int i = 0; i < hw; ++i) {
        dst[4 * i + 0] = saturate_cast(scale[0] * src_c0[i] + bias[0]);
        dst[4 * i + 1] = saturate_cast(scale[1] * src_c1[i] + bias[1]);
        dst[4 * i + 2] = saturate_cast(scale[2] * src_c2[i] + bias[2]);
        if (channel == 4)
            dst[4 * i + 3] = saturate_cast(scale[3] * src_c3[i] + bias[3]);
    }
}

/*
 * Convert a nchw float blob to grayscale uint8 image
 * input blob must have only 1 channel
 */
static void BlobToGray(const float *src, uint8_t *dst, float scale, float bias, int hw) {
    for (int i = 0; i < hw; ++i) {
        dst[i] = saturate_cast(scale * src[i] + bias);
    }
}

/*
 * Convert a nchw float blob to bgr uint8 image
 * input blob must have 3 channel
 */
static void BlobToBGR(const float *src, uint8_t *dst, float *scale, float *bias, int hw) {
    auto src_c0 = src, src_c1 = src + hw, src_c2 = src + hw * 2;
    for (int i = 0; i < hw; ++i) {
        dst[3 * i + 0] = saturate_cast(scale[0] * src_c0[i] + bias[0]);
        dst[3 * i + 1] = saturate_cast(scale[1] * src_c1[i] + bias[1]);
        dst[3 * i + 2] = saturate_cast(scale[2] * src_c2[i] + bias[2]);
    }
}

static void NV12ToBGR(const unsigned char* nv12, unsigned char* bgr, int h, int w) {
    NaiveYUVToBGROrBGRA(nv12, bgr, 3, h, w, true);
}

static void NV21ToBGR(const unsigned char* nv21, unsigned char* bgr, int h, int w) {
    NaiveYUVToBGROrBGRA(nv21, bgr, 3, h, w, false);
}

/*
reverse channel in format rgb uint8
*/
void RGBChannelReverseNaive(uint8_t *src, uint8_t *dst, int channel, int hw) {
    for (int i = 0; i < hw; i++) {
        uint8_t tmp    = src[i * 3];
        dst[i * 3]     = src[i * 3 + 2];
        dst[i * 3 + 2] = tmp;
        dst[i * 3 + 1] = src[i * 3 + 1];
    }
}

/*
reverse channel in format rgba uint8, only reverse rgb
*/
void RGBAChannelReverseNaive(uint8_t *src, uint8_t *dst, int channel, int hw) {
    for (int i = 0; i < hw; i++) {
        uint8_t tmp    = src[i * 4];
        dst[i * 4]     = src[i * 4 + 2];
        dst[i * 4 + 2] = tmp;
        dst[i * 4 + 1] = src[i * 4 + 1];
        if (channel == 4)
            dst[i * 4 + 3] = src[i * 4 + 3];
    }
}

#define FREE_INT8_TEMP_DATA()                               \
    if (desc.data_type == DATA_TYPE_INT8 && blob_data) {    \
        delete[] blob_data;                                 \
    }                                                       \

Status DefaultBlobConverterAcc::ConvertToMatAsync(Mat &image, MatConvertParam param, void *command_queue) {
    Status ret = TNN_OK;
    if (blob_ == nullptr) {
        return Status(TNNERR_NULL_PARAM, "input/output blob is null");
    }
    auto blob_data = reinterpret_cast<float *>(blob_->GetHandle().base);
    auto desc      = blob_->GetBlobDesc();
    auto dims      = desc.dims;
    auto hw        = dims[2] * dims[3];

    if (desc.data_type == DATA_TYPE_INT8) {
        if (image.GetMatType() == RESERVED_INT8_TEST) {
            memcpy(image.GetData(), blob_data, DimsVectorUtils::Count(dims));
            return TNN_OK;
        } else {
            auto real_blob_data = new float[dims[0] * dims[1] * dims[2] * dims[3]];
            auto blob_scale = reinterpret_cast<BlobInt8 *>(blob_)->GetIntResource()->scale_handle.force_to<float *>();
            auto scale_len  = reinterpret_cast<BlobInt8 *>(blob_)->GetIntResource()->scale_handle.GetDataCount();
            NaiveDequant(reinterpret_cast<int8_t *>(blob_->GetHandle().base), blob_scale, scale_len, real_blob_data, dims);
            blob_data = real_blob_data;
        }
    } else if (desc.data_type == DATA_TYPE_BFP16) {
        if (image.GetMatType() == RESERVED_BFP16_TEST) {
            memcpy(image.GetData(), blob_data, DimsVectorUtils::Count(dims) * 2);
            return TNN_OK;
        }
    }

    if (image.GetMatType() == NCHW_FLOAT) {
        for (int n = 0; n < dims[0]; n++) {
            NCHWConvert(blob_data + n * dims[1] * hw, reinterpret_cast<float *>(image.GetData()) + n * dims[1] * hw,
                        param.scale.data(), param.bias.data(), dims[1], hw);
        }
    } else if (image.GetMatType() == N8UC4) {
        for (int n = 0; n < dims[0]; n++) {
            BlobToBGRA(blob_data + n * dims[1] * hw, reinterpret_cast<uint8_t *>(image.GetData()) + n * 4 * hw,
                       param.scale.data(), param.bias.data(), dims[1], hw);
        }
    } else if (image.GetMatType() == N8UC3) {
        for (int n = 0; n < dims[0]; n++) {
            BlobToBGR(blob_data + n * 3 * hw, reinterpret_cast<uint8_t *>(image.GetData()) + n * 3 * hw,
                      param.scale.data(), param.bias.data(), hw);
        }
    } else if (image.GetMatType() == NGRAY) {
        for (int n = 0; n < dims[0]; n++) {
            BlobToGray(blob_data + n * hw, reinterpret_cast<uint8_t *>(image.GetData()) + n * hw, param.scale[0],
                       param.bias[0], hw);
        }
    } else if (image.GetMatType() == RESERVED_BFP16_TEST) {
        for (int n = 0; n < DimsVectorUtils::Count(dims); n++) {
            reinterpret_cast<bfp16_t *>(image.GetData())[n] = blob_data[n];
        }
    } else {
        FREE_INT8_TEMP_DATA();
        return Status(TNNERR_PARAM_ERR, "convert type not support yet");
    }

    // reverse channel before convert if needed
    if (param.reverse_channel) {
        if (image.GetMatType() == N8UC3) {
            for (int n = 0; n < dims[0]; n++) {
                RGBChannelReverseNaive(
                    reinterpret_cast<uint8_t *>(image.GetData()) + n * 3 * hw,
                    reinterpret_cast<uint8_t *>(image.GetData()) + n * 3 * hw, dims[1], hw);
            }
        } else if (image.GetMatType() == N8UC4) {
            for (int n = 0; n < dims[0]; n++) {
                RGBAChannelReverseNaive(
                    reinterpret_cast<uint8_t *>(image.GetData()) + n * 4 * hw,
                    reinterpret_cast<uint8_t *>(image.GetData()) + n * 4 * hw, dims[1], hw);
            }
        } else {
            FREE_INT8_TEMP_DATA();
            return Status(TNNERR_PARAM_ERR, "reverse type not support yet, mat type: " +
                          std::to_string(image.GetMatType()));
        }
    }

    FREE_INT8_TEMP_DATA();
    return ret;
}

Status DefaultBlobConverterAcc::ConvertFromMatFunc(Mat& image, float* blob_data,
        MatConvertParam& param, BlobDesc& desc, const DimsVector& dims, const int hw) {
    if (image.GetMatType() == NCHW_FLOAT) {
        for (int n = 0; n < dims[0]; n++) {
            NCHWConvert(reinterpret_cast<float *>(image.GetData()) + n * dims[1] * hw, blob_data + n * dims[1] * hw,
                        param.scale.data(), param.bias.data(), dims[1], hw);
        }
    } else if (image.GetMatType() == N8UC4) {
        for (int n = 0; n < dims[0]; n++) {
            BGRAToBlob(reinterpret_cast<uint8_t *>(image.GetData()) + n * 4 * hw, blob_data + n * dims[1] * hw,
                       param.scale.data(), param.bias.data(), dims[1], hw);
        }
    } else if (image.GetMatType() == N8UC3) {
        for (int n = 0; n < dims[0]; n++) {
            BGRToBlob(reinterpret_cast<uint8_t *>(image.GetData()) + n * 3 * hw, blob_data + n * 3 * hw,
                      param.scale.data(), param.bias.data(), hw);
        }
    } else if (image.GetMatType() == NGRAY) {
        for (int n = 0; n < dims[0]; n++) {
            GrayToBlob(reinterpret_cast<uint8_t *>(image.GetData()) + n * hw, blob_data + n * hw, param.scale[0],
                       param.bias[0], hw);
        }
    } else if (image.GetMatType() == NNV12) {
        Mat bgr(DEVICE_NAIVE, RESERVED_INT8_TEST, image.GetDims());
        for (int n = 0; n < dims[0]; n++) {
            NV12ToBGR(reinterpret_cast<uint8_t *>(image.GetData()) + n * 3 * hw / 2,
                      reinterpret_cast<uint8_t *>(bgr.GetData()) + n * 3 * hw, dims[2], dims[3]);
            BGRToBlob(reinterpret_cast<uint8_t *>(bgr.GetData()) + n * 3 * hw, blob_data + n * 3 * hw,
                      param.scale.data(), param.bias.data(), hw);
        }
    } else if (image.GetMatType() == NNV21) {
        Mat bgr(DEVICE_NAIVE, RESERVED_INT8_TEST, image.GetDims());
        for (int n = 0; n < dims[0]; n++) {
            NV21ToBGR(reinterpret_cast<uint8_t *>(image.GetData()) + n * 3 * hw / 2,
                      reinterpret_cast<uint8_t *>(bgr.GetData()) + n * 3 * hw, dims[2], dims[3]);
            BGRToBlob(reinterpret_cast<uint8_t *>(bgr.GetData()) + n * 3 * hw, blob_data + n * 3 * hw,
                      param.scale.data(), param.bias.data(), hw);
        }
    } else if (image.GetMatType() == RESERVED_BFP16_TEST) {
        for (int n = 0; n < DimsVectorUtils::Count(dims); n++) {
            blob_data[n] = float(reinterpret_cast<bfp16_t *>(image.GetData())[n]);
        }
    } else {
        return Status(TNNERR_PARAM_ERR, "convert type not support yet");
    }

    return TNN_OK;
}

Status DefaultBlobConverterAcc::ConvertFromMatAsync(Mat &image_src, MatConvertParam param, void *command_queue) {
    if (blob_ == nullptr) {
        return Status(TNNERR_NULL_PARAM, "input/output blob_ is null");
    }
    auto desc      = blob_->GetBlobDesc();
    auto dims      = desc.dims;
    auto hw        = dims[2] * dims[3];
    auto blob_data = reinterpret_cast<float *>(blob_->GetHandle().base);
    if (desc.data_type == DATA_TYPE_INT8) {
        if (image_src.GetMatType() == RESERVED_INT8_TEST) {
            memcpy(blob_data, image_src.GetData(), DimsVectorUtils::Count(dims));
            return TNN_OK;
        } else
            blob_data = new float[dims[0] * dims[1] * hw];
    } else if (desc.data_type == DATA_TYPE_BFP16) {
        if (image_src.GetMatType() == RESERVED_BFP16_TEST) {
            memcpy(blob_data, image_src.GetData(), DimsVectorUtils::Count(dims) * 2);
            return TNN_OK;
        }
    }

    Mat image(image_src.GetDeviceType(), image_src.GetMatType(), image_src.GetDims(), image_src.GetData());

    // reverse channel before convert if needed
    if (param.reverse_channel) {
        Mat reversed(image.GetDeviceType(), image.GetMatType(), image.GetDims());
        if (image.GetMatType() == N8UC3) {
            for (int n = 0; n < dims[0]; n++) {
                RGBChannelReverseNaive(
                    reinterpret_cast<uint8_t *>(image.GetData()) + n * 3 * hw,
                    reinterpret_cast<uint8_t *>(reversed.GetData()) + n * 3 * hw, dims[1], hw);
            }
        } else if (image.GetMatType() == N8UC4) {
            for (int n = 0; n < dims[0]; n++) {
                RGBAChannelReverseNaive(
                    reinterpret_cast<uint8_t *>(image.GetData()) + n * 4 * hw,
                    reinterpret_cast<uint8_t *>(reversed.GetData()) + n * 4 * hw, dims[1], hw);
            }
        } else {
            FREE_INT8_TEMP_DATA();
            return Status(TNNERR_PARAM_ERR, "reverse type not support yet, mat type: " +
                          std::to_string(image.GetMatType()));
        }
        image = reversed;
    }

    Status ret = ConvertFromMatFunc(image, blob_data, param, desc, dims, hw);
    if (ret != TNN_OK) {
        FREE_INT8_TEMP_DATA();
        return ret;
    }

    if (desc.data_type == DATA_TYPE_INT8) {
        auto blob_scale     = reinterpret_cast<BlobInt8 *>(blob_)->GetIntResource()->scale_handle.force_to<float *>();
        auto scale_len      = reinterpret_cast<BlobInt8 *>(blob_)->GetIntResource()->scale_handle.GetDataCount();
        auto real_blob_data = reinterpret_cast<int8_t *>(blob_->GetHandle().base);
        NaiveQuant(blob_data, blob_scale, scale_len, real_blob_data, dims);
        delete[] blob_data;
    }
    return TNN_OK;
}

#undef FREE_INT8_TEMP_DATA

Status DefaultBlobConverterAcc::ConvertToMat(Mat &image, MatConvertParam param, void *command_queue) {
    return ConvertToMatAsync(image, param, command_queue);
}

Status DefaultBlobConverterAcc::ConvertFromMat(Mat &image, MatConvertParam param, void *command_queue) {
    return ConvertFromMatAsync(image, param, command_queue);
}

}  // namespace TNN_NS
