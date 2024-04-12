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

#ifndef TNN_SOURCE_TNN_DEVICE_ATLAS_ATLAS_BLOB_CONVERTER_H_
#define TNN_SOURCE_TNN_DEVICE_ATLAS_ATLAS_BLOB_CONVERTER_H_

#include "acl/acl.h"
#include "tnn/core/macro.h"
#include "tnn/device/atlas/atlas_common_types.h"
#include "tnn/utils/blob_converter.h"
#include "tnn/utils/blob_converter_internal.h"

namespace TNN_NS {

typedef enum { AIPP_NONE = 0, AIPP_STATIC, AIPP_DYNAMIC } AippType;

class AtlasBlobConverterAcc : public BlobConverterAcc {
public:
    AtlasBlobConverterAcc(Blob* blob);
    virtual ~AtlasBlobConverterAcc();

    virtual Status ConvertToMat(Mat& mat, MatConvertParam param, void* command_queue = NULL);
    virtual Status ConvertToMatAsync(Mat& mat, MatConvertParam param, void* command_queue = NULL);

    virtual Status ConvertFromMat(Mat& mat, MatConvertParam param, void* command_queue = NULL);
    virtual Status ConvertFromMatAsync(Mat& mat, MatConvertParam param, void* command_queue = NULL);

private:
    Status ConvertFromMatAsyncWithoutAipp(Mat& mat, MatConvertParam param, const aclrtStream& aclrt_stream);
    Status ConvertFromMatAsyncWithStaticAipp(Mat& mat, MatConvertParam param, const aclrtStream& aclrt_stream);
    Status ConvertFromMatAsyncWithDynamicAipp(Mat& mat, MatConvertParam param, const aclrtStream& aclrt_stream);

    bool NeedDoScaleBias(MatConvertParam& param);
    Status AtlasMemoryCopyAsync(void* dst, void* src, DeviceType mat_device_type, int bytes, void* stream,
                                bool from_mat);
    Status SetDynamicAipp(Mat& mat, MatConvertParam& param);

    bool input_blob_info_found_   = false;
    bool do_scale_bias_           = true;
    int blob_bytesize_            = 0;
    std::shared_ptr<char> buffer_ = nullptr;

    aclmdlAIPP* aipp_dynamic_set_ = nullptr;
    AippType aipp_type_           = AIPP_NONE;
    int aipp_mat_batchsize_       = 0;
    size_t dynamic_aipp_index_    = 0;
    std::shared_ptr<AtlasOMModelInfo> om_model_info_;
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_ATLAS_ATLAS_BLOB_CONVERTER_H_
