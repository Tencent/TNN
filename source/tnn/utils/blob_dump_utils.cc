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

#include "tnn/core/mat.h"
#include "tnn/interpreter/layer_resource.h"
#include "tnn/utils/blob_converter.h"
#include "tnn/utils/dims_vector_utils.h"
#include "tnn/utils/string_utils_inner.h"

namespace TNN_NS {

#pragma warning(push)
#pragma warning(disable : 4996)

#if (DUMP_INPUT_BLOB || DUMP_OUTPUT_BLOB)
#ifdef __ANDROID__
std::string g_tnn_dump_directory = "/storage/emulated/0/";
#else
std::string g_tnn_dump_directory = "./";
#endif //__ANDROID__
#endif

// #define DUMP_RAW_INT8

std::string BlobDescToString(BlobDesc desc) {
    std::string dims_info = "dims";
    for(int i = 0; i < desc.dims.size(); ++i) {
       dims_info += "-" + ToString(desc.dims[i]);
    }

    // blob name rather than layer name
    char ss[1000];
    std::string name = desc.name;
    std::replace(name.begin(), name.end(), '/', '_');
    snprintf(ss, 1000, "%s-%s", name.c_str(), dims_info.c_str());
    return std::string(ss);
}

Status DumpDeviceBlob(Blob* blob, Context* context, std::string fname_prefix) {
    void* command_queue;
    context->GetCommandQueue(&command_queue);

    auto blob_desc = blob->GetBlobDesc();
    MatType mat_type = NCHW_FLOAT;
    auto data_type = blob_desc.data_type;

#ifdef DUMP_RAW_INT8
    if(blob_desc.data_type == DATA_TYPE_INT8) {
        mat_type = RESERVED_INT8_TEST;
    }
#endif
    if(blob_desc.data_type == DATA_TYPE_INT32) {
        mat_type = NC_INT32;
    }

    Mat cpu_mat(DEVICE_NAIVE, mat_type, blob_desc.dims);
    void *data_ptr = cpu_mat.GetData();
    
    if (blob->GetBlobDesc().device_type != DEVICE_NAIVE) {
        BlobConverter blob_converter_dev(blob);
        Status ret = blob_converter_dev.ConvertToMat(cpu_mat, MatConvertParam(), command_queue);
        if (ret != TNN_OK) {
            LOGE("output blob_converter failed (%s)\n", ret.description().c_str());
            return ret;
        }
    } else {
        data_ptr = blob->GetHandle().base;
    }
    
    char fname[1000];
    snprintf(fname, 1000, "%s-%s.txt", fname_prefix.c_str(), BlobDescToString(blob_desc).c_str());
    FILE* fp = fopen(fname, "wb");
    if (!fp) {
        return Status(TNNERR_OPEN_FILE, "open file error");
    }
    int count = DimsVectorUtils::Count(blob_desc.dims);

#ifdef DUMP_RAW_INT8
    int8_t* ptr = reinterpret_cast<int8_t*>(cpu_mat.GetData());
    for (int n = 0; n < count; ++n) {
        fprintf(fp, "%d\n", int(ptr[n]));
    }
#else
    if (data_type == DATA_TYPE_FLOAT) {
        auto ptr = (float *)data_ptr;
        for (int n = 0; n < count; ++n) {
            fprintf(fp, "%.9f\n", ptr[n]);
        }
    } else if (data_type == DATA_TYPE_INT32) {
        auto ptr = (int *)data_ptr;
        for (int n = 0; n < count; ++n) {
            fprintf(fp, "%d\n", ptr[n]);
        }
    } else if (data_type == DATA_TYPE_UINT32) {
        auto ptr = (unsigned int *)data_ptr;
        for (int n = 0; n < count; ++n) {
            fprintf(fp, "%d\n", ptr[n]);
        }
    } else {
        LOGE("unsupport data type to dump: %d\n", data_type);
    }
#endif

    fclose(fp);

    return TNN_OK;
}


#pragma warning(push)

}  // namespace TNN_NS
