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

#include "tnn/device/atlas/atlas_blob_converter.h"
#include "acl/acl.h"
#include "tnn/core/macro.h"
#include "tnn/device/atlas/atlas_common_types.h"
#include "tnn/device/atlas/atlas_utils.h"
#include "tnn/memory_manager/blob_memory_size_info.h"
#include "tnn/utils/blob_memory_size_utils.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

// default contructor will create convert buffer
AtlasBlobConverterAcc::AtlasBlobConverterAcc(Blob *blob) : BlobConverterAcc(blob) {
    BlobMemorySizeInfo size_info = Calculate1DMemorySize(blob->GetBlobDesc());
    blob_bytesize_               = GetBlobMemoryBytesSize(size_info);
    LOGD("blob bytesize: %d\n", blob_bytesize_);
}

AtlasBlobConverterAcc::~AtlasBlobConverterAcc() {}

// convert blob data to mat async
Status AtlasBlobConverterAcc::ConvertToMatAsync(Mat &mat, MatConvertParam param, void *command_queue) {
    do_scale_bias_ = NeedDoScaleBias(param);

    if (do_scale_bias_) {
        return Status(TNNERR_PARAM_ERR, "not support postprocess yet!");
    }

    if (mat.GetMatType() != NCHW_FLOAT) {
        return Status(TNNERR_PARAM_ERR, "not support this type convert yet!");
    }

    auto atlas_cmd_queue = static_cast<AtlasCommandQueue *>(command_queue);
    if (atlas_cmd_queue == nullptr) {
        LOGE("get atlas command queue failed!\n");
        return Status(TNNERR_NULL_PARAM, "get atlas command queue failed!");
    }

    aclError ret = aclrtSetCurrentContext(atlas_cmd_queue->context);
    if (ret != ACL_ERROR_NONE) {
        LOGE("set context failed\n");
        return Status(TNNERR_ATLAS_RUNTIME_ERROR, "set context failed");
    }

    DataFormat blob_datatype = blob_->GetBlobDesc().data_format;
    if ((NCHW_FLOAT == mat.GetMatType() && DATA_FORMAT_NCHW == blob_datatype)) {
        if (mat.GetDeviceType() == DEVICE_ATLAS) {
            // need to copy form device to device
            ret = aclrtMemcpyAsync(mat.GetData(), blob_bytesize_, blob_->GetHandle().base, blob_bytesize_,
                    ACL_MEMCPY_DEVICE_TO_DEVICE, atlas_cmd_queue->stream);
            if (ret != ACL_ERROR_NONE) {
                return Status(TNNERR_ATLAS_RUNTIME_ERROR, "acl memory copy failed");
            }
        } else if (mat.GetDeviceType() == DEVICE_NAIVE) {
            // need to copy form device to host
            ret = aclrtMemcpyAsync(mat.GetData(), blob_bytesize_, blob_->GetHandle().base, blob_bytesize_,
                    ACL_MEMCPY_DEVICE_TO_HOST, atlas_cmd_queue->stream);
            if (ret != ACL_ERROR_NONE) {
                return Status(TNNERR_ATLAS_RUNTIME_ERROR, "acl memory copy failed");
            }
        } else {
            return Status(TNNERR_PARAM_ERR, "not support this device type convert yet!");
        }
    } else {
        return Status(TNNERR_PARAM_ERR, "not support this dataformat type convert yet!");
    }

    return TNN_OK;
}

// convert mat data to blob async
Status AtlasBlobConverterAcc::ConvertFromMatAsync(Mat &mat, MatConvertParam param, void *command_queue) {
    do_scale_bias_ = NeedDoScaleBias(param);

    if (do_scale_bias_) {
        return Status(TNNERR_PARAM_ERR, "not support preprocess yet!");
    }

    if (mat.GetMatType() != NCHW_FLOAT) {
        return Status(TNNERR_PARAM_ERR, "not support this type convert yet!");
    }

    auto atlas_cmd_queue = static_cast<AtlasCommandQueue *>(command_queue);
    if (atlas_cmd_queue == nullptr) {
        LOGE("get atlas command queue failed!\n");
        return Status(TNNERR_NULL_PARAM, "get atlas command queue failed!");
    }

    aclError ret = aclrtSetCurrentContext(atlas_cmd_queue->context);
    if (ret != ACL_ERROR_NONE) {
        LOGE("set context failed\n");
        return Status(TNNERR_ATLAS_RUNTIME_ERROR, "set context failed");
    }

    DataFormat blob_datatype = blob_->GetBlobDesc().data_format;
    if ((NCHW_FLOAT == mat.GetMatType() && DATA_FORMAT_NCHW == blob_datatype)) {
        if (mat.GetDeviceType() == DEVICE_ATLAS) {
            // need to copy from device to device
            ret = aclrtMemcpyAsync(blob_->GetHandle().base, blob_bytesize_, mat.GetData(), blob_bytesize_,
                    ACL_MEMCPY_DEVICE_TO_DEVICE, atlas_cmd_queue->stream);
            if (ret != ACL_ERROR_NONE) {
                return Status(TNNERR_ATLAS_RUNTIME_ERROR, "acl memory copy failed");
            }
        } else if (mat.GetDeviceType() == DEVICE_NAIVE) {
            // need to copy from host to device
            ret = aclrtMemcpyAsync(blob_->GetHandle().base, blob_bytesize_, mat.GetData(), blob_bytesize_,
                    ACL_MEMCPY_HOST_TO_DEVICE, atlas_cmd_queue->stream);
            if (ret != ACL_ERROR_NONE) {
                return Status(TNNERR_ATLAS_RUNTIME_ERROR, "acl memory copy failed");
            }
        } else {
            return Status(TNNERR_PARAM_ERR, "not support this device type convert yet!");
        }
    } else {
        return Status(TNNERR_PARAM_ERR, "not support this dataformat type convert yet!");
    }

    return TNN_OK;
}

Status AtlasBlobConverterAcc::ConvertToMat(Mat &mat, MatConvertParam param, void *command_queue) {
    Status ret = ConvertToMatAsync(mat, param, command_queue);
    if (ret == TNN_OK) {
        auto atlas_cmd_queue = static_cast<AtlasCommandQueue *>(command_queue);
        if (atlas_cmd_queue == nullptr) {
            LOGE("get atlas command queue failed!\n");
            return Status(TNNERR_NULL_PARAM, "get atlas command queue failed!");
        }
        aclError acl_ret = aclrtSynchronizeStream(atlas_cmd_queue->stream);
        if (acl_ret != ACL_ERROR_NONE) {
            return Status(TNNERR_ATLAS_RUNTIME_ERROR, "stream sync failed");
        }
    }
    return ret;
}

Status AtlasBlobConverterAcc::ConvertFromMat(Mat &mat, MatConvertParam param, void *command_queue) {
    Status ret = ConvertFromMatAsync(mat, param, command_queue);
    if (ret == TNN_OK) {
        auto atlas_cmd_queue = static_cast<AtlasCommandQueue *>(command_queue);
        if (atlas_cmd_queue == nullptr) {
            LOGE("get atlas command queue failed!\n");
            return Status(TNNERR_NULL_PARAM, "get atlas command queue failed!");
        }
        aclError acl_ret = aclrtSynchronizeStream(atlas_cmd_queue->stream);
        if (acl_ret != ACL_ERROR_NONE) {
            return Status(TNNERR_ATLAS_RUNTIME_ERROR, "stream sync failed");
        }
    }
    return ret;
}

bool AtlasBlobConverterAcc::NeedDoScaleBias(MatConvertParam &param) {
    for (auto s : param.scale) {
        if (s != 1.0f) {
            return true;
        }
    }
    for (auto b : param.bias) {
        if (b != 0.0f) {
            return true;
        }
    }

    return false;
}

DECLARE_BLOB_CONVERTER_CREATER(Atlas);
REGISTER_BLOB_CONVERTER(Atlas, DEVICE_ATLAS);

}  // namespace TNN_NS
