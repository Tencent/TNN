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

#include "tnn/utils/blob_dump_utils.h"

#include <stdlib.h>

#include <algorithm>

#include "tnn/core/blob_int8.h"
#include "tnn/interpreter/layer_resource.h"
#include "tnn/utils/blob_transfer_utils.h"
#include "tnn/utils/data_format_converter.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_vector_utils.h"
#include "tnn/utils/half_utils.h"
#include "tnn/utils/bfp16_utils.h"

namespace TNN_NS {

#pragma warning(push)
#pragma warning(disable : 4996)

using float32x4 = float[4];

#if (DUMP_INPUT_BLOB || DUMP_OUTPUT_BLOB)
std::string g_tnn_dump_directory = "/storage/emulated/0/";
#endif

// define DUMP_RAW_INT8
int dump_ncdhw_float_blob(BlobDesc desc, std::string fname, float* ptr) {
    FILE* fp = fopen(fname.c_str(), "wb");
    if (!fp) {
        LOGE("fopen failed: %s", fname.c_str());
        return -1;
    }

    const int count = DimsVectorUtils::Count(desc.dims);
    LOGD("fname:%s count:%d\n", fname.c_str(), count);

    for (int index = 0; index < count; index++) {
        fprintf(fp, "%.6f\n", ptr[index]);
    }

    fclose(fp);
    return 0;
}

// dump int8 blob to the specified file
int dump_ncdhw_int8_blob(BlobDesc desc, std::string fname, int8_t* ptr) {
    FILE* fp = fopen(fname.c_str(), "wb");
    if (!fp) {
        LOGE("fopen failed: %s", fname.c_str());
        return -1;
    }

    const int count = DimsVectorUtils::Count(desc.dims);
    LOGD("fname:%s count:%d\n", fname.c_str(), count);

    for (int index = 0; index < count; index++) {
        fprintf(fp, "%d\n", ptr[index]);
    }

    fclose(fp);
    return 0;
}

// dump nc4hw4 blob to the specified file in nchw format
int dump_nc4hw4_float_blob(BlobDesc desc, std::string fname, float* ptr) {
    FILE* fp = fopen(fname.c_str(), "wb");
    if (!fp) {
        LOGE("fopen failed: %s", fname.c_str());
        return -1;
    }

    int num     = desc.dims[0];
    int channel = desc.dims[1];
    int height  = desc.dims[2];
    int width   = desc.dims[3];

    // 4 channels packed togather
    int channel_4 = UP_DIV(channel, 4);

    const int count = DimsVectorUtils::Count(desc.dims);
    LOGD("fname:%s count:%d\n", fname.c_str(), count);

    float32x4* ptr_nc4hw4 = (float32x4*)ptr;
    for (int n = 0; n < num; ++n) {
        for (int c = 0; c < channel; ++c) {
            int c_4      = c / 4;
            int c_remain = c % 4;
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    int idx = n * channel_4 * height * width + c_4 * height * width + h * width + w;
                    fprintf(fp, "%.6f\n", ptr_nc4hw4[idx][c_remain]);
                }
            }
        }
    }

    fclose(fp);
    return 0;
}

// dump nhwc4 blob to the specified file in nchw format
int dump_nhwc4_int8_blob(BlobDesc desc, std::string fname, int8_t* ptr) {
    FILE* fp = fopen(fname.c_str(), "wb");
    if (!fp) {
        LOGE("fopen failed: %s", fname.c_str());
        return -1;
    }

    int num     = desc.dims[0];
    int channel = desc.dims[1];
    int height  = desc.dims[2];
    int width   = desc.dims[3];

    // packed 4 channels
    int channel_4 = ROUND_UP(channel, 4);

    const int count = DimsVectorUtils::Count(desc.dims);
    LOGD("fname:%s count:%d\n", fname.c_str(), count);

    for (int n = 0; n < num; ++n) {
        for (int c = 0; c < channel; ++c) {
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    int idx = (n * height * width + h * width + w) * channel_4 + c;
                    fprintf(fp, "%d\n", ptr[idx]);
                }
            }
        }
    }

    fclose(fp);
    return 0;
}

// dump nchw blob to the specified file in nchw format
int dump_nchw_float_blob(BlobDesc desc, std::string fname, float* ptr) {
    FILE* fp = fopen(fname.c_str(), "wb");
    if (!fp)
        return -1;

    int num     = desc.dims[0];
    int channel = desc.dims[1];
    int height  = desc.dims[2];
    int width   = desc.dims[3];

    int size = num * channel * height * width;
    LOGD("fname:%s size:%d\n", fname.c_str(), size);

    for (int n = 0; n < num; ++n) {
        for (int c = 0; c < channel; ++c) {
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    int idx = n * channel * height * width + c * height * width + h * width + w;
                    fprintf(fp, "%.9f\n", ptr[idx]);
                }
            }
        }
    }
    fclose(fp);
    return 0;
}

std::string BlobDescToString(BlobDesc desc) {
    char dim[1000];
    if (desc.dims.size() == 5) {
        snprintf(dim, 1000, "NCDHW-%d-%d-%d-%d-%d", desc.dims[0], desc.dims[1], desc.dims[2], desc.dims[3],
                 desc.dims[4]);
    } else {
        snprintf(dim, 1000, "NCHW-%d-%d-%d-%d", desc.dims[0], desc.dims[1], desc.dims[2], desc.dims[3]);
    }
    // blob name rather than layer name
    char ss[1000];
    std::string name = desc.name;
    std::replace(name.begin(), name.end(), '/', '_');
    snprintf(ss, 1000, "%s-%s", name.c_str(), dim);
    return std::string(ss);
}

void DumpBlobData(float* cpu_data, Blob* dev_blob, char *fname, BlobDesc& dev_blob_desc, int &ret_code) {
    switch (dev_blob_desc.data_format) {
        case DATA_FORMAT_NCDHW:
        case DATA_FORMAT_NCHW:
#ifdef DUMP_RAW_INT8
            /*
             * dump the int8 data with out dequantization.
             * this option is used for debug.
             */
            if (dev_blob_desc.data_type == DATA_TYPE_INT8) {
                ret_code = dump_ncdhw_int8_blob(dev_blob->GetBlobDesc(), std::string(fname), (int8_t*)cpu_data);
            } else
#endif
            {
                ret_code = dump_ncdhw_float_blob(dev_blob->GetBlobDesc(), std::string(fname), cpu_data);
            }
            break;
        case DATA_FORMAT_NC4HW4:
        case DATA_FORMAT_NHWC4: {
#ifdef DUMP_RAW_INT8
            /*
             * dump the int8 data with out dequantization.
             * this option is used for debug.
             */
            if (dev_blob_desc.data_type == DATA_TYPE_INT8) {
                ret_code = dump_nhwc4_int8_blob(dev_blob->GetBlobDesc(), std::string(fname), (int8_t*)cpu_data);
            } else
#endif
            {
                ret_code = dump_nc4hw4_float_blob(dev_blob->GetBlobDesc(), std::string(fname), cpu_data);
            }
        } break;
        default:
            break;
    }
}

/*
 * device blob dump does the following things:
 *  1. copy
 *  2. convert format and data type
 *  3. dump to file
 */
Status DumpDeviceBlob(Blob* dev_blob, Context* context, std::string fname_prefix) {
    int data_count     = 0;
    auto dev_blob_desc = dev_blob->GetBlobDesc();
    if (dev_blob_desc.data_format == DATA_FORMAT_NC4HW4 || dev_blob_desc.data_format == DATA_FORMAT_NHWC4) {
        data_count = dev_blob_desc.dims[0] * ROUND_UP(dev_blob_desc.dims[1], 4) *
                     ROUND_UP(dev_blob_desc.dims[2] * dev_blob_desc.dims[3], 4);
    } else {
        data_count = DimsVectorUtils::Count(dev_blob_desc.dims);
    }

    size_t size_in_bytes = data_count * DataTypeUtils::GetBytesSize(dev_blob_desc.data_type);

    LOGD("cpu blob size in bytes:%lu\n", size_in_bytes);

    std::shared_ptr<char> cpu_ptr(new char[size_in_bytes]);
    BlobHandle cpu_handle;
    cpu_handle.base                = reinterpret_cast<void*>(cpu_ptr.get());
    std::shared_ptr<Blob> cpu_blob = std::make_shared<Blob>(dev_blob->GetBlobDesc(), cpu_handle);

    // step 1. copy from device to cpu
    void* command_queue;
    context->GetCommandQueue(&command_queue);
    auto ret = CopyFromDevice(cpu_blob.get(), dev_blob, command_queue);
    if (ret != TNN_OK) {
        LOGD("copy blob from device failed\n");
        return ret;
    }

    // step 2. convert data_type
    // convert to float
    float* cpu_data = (float*)((char*)cpu_handle.base + cpu_handle.bytes_offset);
    std::shared_ptr<float> convert_ptr;

    if (dev_blob_desc.data_type == DATA_TYPE_HALF) {
        cpu_data    = new float[data_count];
        convert_ptr = std::shared_ptr<float>(cpu_data);
        ConvertFromHalfToFloat((void*)((char*)cpu_handle.base + cpu_handle.bytes_offset), cpu_data, data_count);
    } else if (dev_blob_desc.data_type == DATA_TYPE_BFP16) {
        cpu_data    = new float[data_count];
        convert_ptr = std::shared_ptr<float>(cpu_data);
        ConvertFromBFP16ToFloat((void*)((char*)cpu_handle.base + cpu_handle.bytes_offset), cpu_data, data_count);
    } else if (dev_blob_desc.data_type == DATA_TYPE_INT8) {
#ifndef DUMP_RAW_INT8
        IntScaleResource* re = reinterpret_cast<BlobInt8*>(dev_blob)->GetIntResource();
        if (!re || !re->scale_handle.GetDataCount()) {
            return TNN_OK;
        }
        cpu_data    = new float[data_count];
        convert_ptr = std::shared_ptr<float>(cpu_data);
        switch (dev_blob_desc.data_format) {
            case DATA_FORMAT_NCDHW:
            case DATA_FORMAT_NCHW:
                DataFormatConverter::ConvertFromInt8ToFloatNCHW(
                    (int8_t*)cpu_handle.base + cpu_handle.bytes_offset, cpu_data, re->scale_handle.force_to<float*>(),
                    re->scale_handle.GetDataCount(), dev_blob_desc.dims[0], dev_blob_desc.dims[1],
                    dev_blob_desc.dims[2], dev_blob_desc.dims[3]);
                break;
            case DATA_FORMAT_NC4HW4:
                DataFormatConverter::ConvertFromInt8ToFloatNCHW4(
                    (int8_t*)cpu_handle.base + cpu_handle.bytes_offset, cpu_data, re->scale_handle.force_to<float*>(),
                    re->scale_handle.GetDataCount(), dev_blob_desc.dims[0], dev_blob_desc.dims[1],
                    dev_blob_desc.dims[2], dev_blob_desc.dims[3]);
                break;

            case DATA_FORMAT_NHWC4:
                DataFormatConverter::ConvertFromInt8ToFloatNHWC4(
                    (int8_t*)cpu_handle.base + cpu_handle.bytes_offset, cpu_data, re->scale_handle.force_to<float*>(),
                    re->scale_handle.GetDataCount(), dev_blob_desc.dims[0], dev_blob_desc.dims[1],
                    dev_blob_desc.dims[2], dev_blob_desc.dims[3]);
                break;

            default:
                return Status(TNNERR_PARAM_ERR, "unsupport data format");
        }
#endif
    }

    // step 3. convert format and dump to file
    char fname[1000];
    snprintf(fname, 1000, "%s-%s.txt", fname_prefix.c_str(), BlobDescToString(cpu_blob->GetBlobDesc()).c_str());

    int ret_code = 0;
    DumpBlobData(cpu_data, dev_blob, fname, dev_blob_desc, ret_code);

    if (ret_code != 0) {
        LOGE("dump blob error\n");
        return Status(TNNERR_PARAM_ERR, "dump blob error");
    }

    return TNN_OK;
}

#pragma warning(push)

}  // namespace TNN_NS
