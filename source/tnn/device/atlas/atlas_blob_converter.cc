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

#include "tnn/core/macro.h"
#include "tnn/device/atlas/atlas_runtime.h"
#include "tnn/device/atlas/atlas_utils.h"
#include "tnn/memory_manager/blob_memory_size_info.h"
#include "tnn/utils/blob_memory_size_utils.h"
#include "tnn/utils/data_format_converter.h"
#include "tnn/utils/dims_vector_utils.h"
#include "tnn/utils/mat_utils.h"

namespace TNN_NS {

// default contructor will create convert buffer
AtlasBlobConverterAcc::AtlasBlobConverterAcc(Blob *blob) : BlobConverterAcc(blob) {
    BlobMemorySizeInfo size_info = Calculate1DMemorySize(blob->GetBlobDesc());
    blob_bytesize_               = GetBlobMemoryBytesSize(size_info);
    LOGD("blob bytesize: %d\n", blob_bytesize_);

    auto model_info_map = AtlasRuntime::GetInstance()->GetModleInfoMap();
    // for input blob, need to find model info
    if (model_info_map.find(blob) != model_info_map.end()) {
        model_info_ = model_info_map[blob];
        aclError acl_ret =
            aclmdlGetInputIndexByName(model_info_.model_desc, ACL_DYNAMIC_AIPP_NAME, &dynamic_aipp_index_);
        LOGD("acl ret: %d  input_index: %d\n", acl_ret, dynamic_aipp_index_);
        if (ACL_ERROR_NONE == acl_ret) {
            aipp_type_ = AIPP_DYNAMIC;
        } else {
            if (model_info_.has_aipp) {
                aipp_type_ = AIPP_STATIC;
            } else {
                aipp_type_ = AIPP_NONE;
            }
        }
        input_blob_info_found_ = true;
    } else {
        input_blob_info_found_ = false;
    }
}

AtlasBlobConverterAcc::~AtlasBlobConverterAcc() {
    if (nullptr != aipp_dynamic_set_) {
        aclError ret = aclmdlDestroyAIPP(aipp_dynamic_set_);
        if (ret != ACL_ERROR_NONE) {
            LOGE("destory aipp_dynamic_set falied\n");
        }
    }
}

// convert blob data to mat async
Status AtlasBlobConverterAcc::ConvertToMatAsync(Mat &mat, MatConvertParam param, void *command_queue) {
    Status tnn_ret   = TNN_OK;
    aclError acl_ret = ACL_ERROR_NONE;

    do_scale_bias_ = NeedDoScaleBias(param);

    if (do_scale_bias_) {
        return Status(TNNERR_PARAM_ERR, "not support postprocess yet!");
    }

    auto atlas_cmd_queue = static_cast<AtlasCommandQueue *>(command_queue);
    if (atlas_cmd_queue == nullptr) {
        LOGE("get atlas command queue failed!\n");
        return Status(TNNERR_NULL_PARAM, "get atlas command queue failed!");
    }

    acl_ret = aclrtSetCurrentContext(atlas_cmd_queue->context);
    if (acl_ret != ACL_ERROR_NONE) {
        LOGE("set context failed\n");
        return Status(TNNERR_ATLAS_RUNTIME_ERROR, "set context failed");
    }

    DataFormat blob_dataformat = blob_->GetBlobDesc().data_format;
    DataType blob_datatype     = blob_->GetBlobDesc().data_type;

    // For Dynamic Models, Output Shape may have changed, get new Output Blob Shape.
    BlobMemorySizeInfo new_size_info = Calculate1DMemorySize(blob_->GetBlobDesc());
    blob_bytesize_                   = GetBlobMemoryBytesSize(new_size_info);
    if (NCHW_FLOAT == mat.GetMatType()) {
        LOGD("Convert To Mat:  mat type: %d, mat device type: %d, byte_size: %d.\n", mat.GetMatType(), mat.GetDeviceType(), blob_bytesize_);
        if (DATA_FORMAT_NCHW == blob_dataformat && DATA_TYPE_FLOAT == blob_datatype) {
            tnn_ret = AtlasMemoryCopyAsync(mat.GetData(), blob_->GetHandle().base, mat.GetDeviceType(), blob_bytesize_,
                                           atlas_cmd_queue->stream, false);
            if (tnn_ret != TNN_OK)
                return tnn_ret;
        } else if (DATA_FORMAT_NHWC == blob_dataformat && DATA_TYPE_FLOAT == blob_datatype) {
            // only support DEVICE_NAIVE device type
            if (DEVICE_NAIVE == mat.GetDeviceType()) {
                if (nullptr == buffer_) {
                    buffer_.reset(new char[blob_bytesize_], [](char *p) { delete[] p; });
                }
                tnn_ret = AtlasMemoryCopyAsync(buffer_.get(), blob_->GetHandle().base, DEVICE_NAIVE, blob_bytesize_,
                                               atlas_cmd_queue->stream, false);
                if (tnn_ret != TNN_OK)
                    return tnn_ret;
                // force sync
                LOGD("force sync to get buffer data\n");
                acl_ret = aclrtSynchronizeStream(atlas_cmd_queue->stream);
                if (acl_ret != ACL_ERROR_NONE) {
                    return Status(TNNERR_ATLAS_RUNTIME_ERROR, "stream sync failed");
                }
                LOGD("convert from nhwc to nchw\n");
                auto blob_dim = blob_->GetBlobDesc().dims;
                DataFormatConverter::ConvertFromNHWCToNCHWFloat((float *)buffer_.get(), (float *)mat.GetData(),
                                                                blob_dim[0], blob_dim[3], blob_dim[1], blob_dim[2]);
            } else {
                return Status(TNNERR_PARAM_ERR, "not support this device type convert yet!");
            }
        } else if ((DATA_FORMAT_NCHW == blob_dataformat || DATA_FORMAT_NHWC == blob_dataformat) && DATA_TYPE_INT64 == blob_datatype) {
            if (DEVICE_NAIVE == mat.GetDeviceType()) {
                if (nullptr == buffer_) {
                    buffer_.reset(new char[blob_bytesize_], [](char *p) { delete[] p; });
                }
                tnn_ret = AtlasMemoryCopyAsync(buffer_.get(), blob_->GetHandle().base, DEVICE_NAIVE, blob_bytesize_,
                                               atlas_cmd_queue->stream, false);
                if (tnn_ret != TNN_OK)
                    return tnn_ret;
                // force sync
                LOGD("force sync to get buffer data\n");
                acl_ret = aclrtSynchronizeStream(atlas_cmd_queue->stream);
                if (acl_ret != ACL_ERROR_NONE) {
                    return Status(TNNERR_ATLAS_RUNTIME_ERROR, "stream sync failed");
                }
                LOGD("convert from int64 to fp32\n");
                auto blob_dim = blob_->GetBlobDesc().dims;
                if (DATA_FORMAT_NCHW == blob_dataformat) {
                    DataFormatConverter::ConvertFromInt64ToFloatNCHW((int64_t *)buffer_.get(), (float *)mat.GetData(),
                                                                     blob_dim[0], blob_dim[3], blob_dim[1], blob_dim[2]);
                } else if (DATA_FORMAT_NHWC == blob_dataformat) {
                    DataFormatConverter::ConvertFromInt64NHWCToFloatNCHW((int64_t *)buffer_.get(), (float *)mat.GetData(),
                                                                         blob_dim[0], blob_dim[3], blob_dim[1], blob_dim[2]);
                } else {
                    return Status(TNNERR_PARAM_ERR, "not support this data format convert yet!");
                }
            } else {
                return Status(TNNERR_PARAM_ERR, "not support this device type convert yet!");
            }
        } else {
            char error_msg[256];
            sprintf(error_msg, "not support this dataformat type convert yet! (data format: %d  data type: %d)", blob_dataformat, blob_datatype);
            return Status(TNNERR_PARAM_ERR, error_msg);
        }
    } else if (NC_INT32 == mat.GetMatType()) {
        LOGD("Convert To NC_INT32 Mat: mat type: %d, mat device type: %d, byte_size: %d.\n", mat.GetMatType(), mat.GetDeviceType(), blob_bytesize_);
        if (DATA_TYPE_INT32 == blob_datatype) {
            tnn_ret = AtlasMemoryCopyAsync(mat.GetData(), blob_->GetHandle().base, mat.GetDeviceType(), blob_bytesize_,
                                           atlas_cmd_queue->stream, false);
            if (tnn_ret != TNN_OK)
                return tnn_ret;
        } else if (DATA_TYPE_FLOAT == blob_datatype) {
            LOGD("WARNING: Target Blob name is '%s', internally convert Blob DataType from FLOAT to INT32.", blob_->GetBlobDesc().name.c_str());
            blob_->GetBlobDesc().data_type = DATA_TYPE_INT32;
            tnn_ret = AtlasMemoryCopyAsync(mat.GetData(), blob_->GetHandle().base, mat.GetDeviceType(), blob_bytesize_,
                                           atlas_cmd_queue->stream, false);
            if (tnn_ret != TNN_OK)
                return tnn_ret;
        } else {
            LOGE("Convert To NC_INT32 Mat: target blob Should be INT32, or FLOAT(convert to INT32)but got %d: %d\n", int(blob_datatype));
            return Status(TNNERR_PARAM_ERR, "Convert To NC_INT32 Mat: target blob Should be INT32.");
        }
    } else if (NC_INT64 == mat.GetMatType()) {
        LOGD("Convert To NC_INT64 Mat: mat type: %d, mat device type: %d, byte_size: %d.\n", mat.GetMatType(), mat.GetDeviceType(), blob_bytesize_);
        if (DATA_TYPE_INT64 == blob_datatype) {
            tnn_ret = AtlasMemoryCopyAsync(mat.GetData(), blob_->GetHandle().base, mat.GetDeviceType(), blob_bytesize_,
                                           atlas_cmd_queue->stream, false);
            if (tnn_ret != TNN_OK)
                return tnn_ret;
        } else if (DATA_TYPE_FLOAT == blob_datatype || DATA_TYPE_INT32 == blob_datatype) {
            LOGD("WARNING: Target Blob name is '%s', internally convert Blob DataType from FLOAT/INT32 to INT64, re-calculate blob size.", blob_->GetBlobDesc().name.c_str());
            blob_->GetBlobDesc().data_type   = DATA_TYPE_INT64;
            BlobMemorySizeInfo new_size_info = Calculate1DMemorySize(blob_->GetBlobDesc());
            blob_bytesize_                   = GetBlobMemoryBytesSize(new_size_info);  // sizeof(int64_t) == 8, re-calculate ByteSize
            tnn_ret = AtlasMemoryCopyAsync(mat.GetData(), blob_->GetHandle().base, mat.GetDeviceType(), blob_bytesize_,
                                           atlas_cmd_queue->stream, false);
            if (tnn_ret != TNN_OK)
                return tnn_ret;
        } else {
            LOGE("Convert To NC_INT64 Mat: target blob Should be INT64, or FLOAT/INT32(convert to INT64)but got %d: %d\n", int(blob_datatype));
            return Status(TNNERR_PARAM_ERR, "Convert To NC_INT64 Mat: target blob Should be INT64.");
        }
    } else {
        return Status(TNNERR_PARAM_ERR, "not support this mat type convert yet!");
    }

    return TNN_OK;
}

// convert mat data to blob async
Status AtlasBlobConverterAcc::ConvertFromMatAsync(Mat &mat, MatConvertParam param, void *command_queue) {
    if (!input_blob_info_found_) {
        LOGE("blob converter init failed, input_blob not found in model info map!\n");
        return Status(TNNERR_COMMON_ERROR, "blob converter init failed, input_blob not found in model info map!");
    }

    Status tnn_ret   = TNN_OK;
    aclError acl_ret = ACL_ERROR_NONE;

    auto atlas_cmd_queue = static_cast<AtlasCommandQueue *>(command_queue);
    if (atlas_cmd_queue == nullptr) {
        LOGE("get atlas command queue failed!\n");
        return Status(TNNERR_NULL_PARAM, "get atlas command queue failed!");
    }

    acl_ret = aclrtSetCurrentContext(atlas_cmd_queue->context);
    if (acl_ret != ACL_ERROR_NONE) {
        LOGE("set context failed\n");
        return Status(TNNERR_ATLAS_RUNTIME_ERROR, "set context failed");
    }

    if (AIPP_DYNAMIC == aipp_type_) {
        LOGD("run with dynamic aipp\n");
        tnn_ret = ConvertFromMatAsyncWithDynamicAipp(mat, param, atlas_cmd_queue);
    } else if (AIPP_STATIC == aipp_type_) {
        LOGD("run with static aipp\n");
        tnn_ret = ConvertFromMatAsyncWithStaticAipp(mat, param, atlas_cmd_queue);
    } else {
        LOGD("run without aipp\n");
        tnn_ret = ConvertFromMatAsyncWithoutAipp(mat, param, atlas_cmd_queue);
    }

    return tnn_ret;
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
    if (!input_blob_info_found_) {
        LOGE("blob converter init failed, input_blob not found in model info map!\n");
        return Status(TNNERR_COMMON_ERROR, "blob converter init failed, input_blob not found in model info map!");
    }

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

Status AtlasBlobConverterAcc::ConvertFromMatAsyncWithoutAipp(Mat &mat, MatConvertParam param,
                                                             AtlasCommandQueue *atlas_cmd_queue) {
    Status tnn_ret   = TNN_OK;
    aclError acl_ret = ACL_ERROR_NONE;

    do_scale_bias_ = NeedDoScaleBias(param);

    if (do_scale_bias_) {
        LOGE("warning: mat convert param is useless in no-dynamic aipp model!\n");
    }

    int mat_bytesize = 0;
    tnn_ret          = MatUtils::GetMatByteSize(mat, mat_bytesize);
    if (TNN_OK != tnn_ret) {
        LOGE("GetMatByteSize failed in ConvertFromMatAsyncWithoutAipp\n");
        return tnn_ret;
    }

    // For input with one or more dim == 0, no need to call ATLAS Memcpy
    if (mat_bytesize == 0) {
        LOGD("Convert From Mat, blob_name = '%s', Blob Size = 0, Skip AtlasMemcpyHostToDevice Step.\n", blob_->GetBlobDesc().name.c_str());
        return TNN_OK;
    }

    DataFormat blob_dataformat = blob_->GetBlobDesc().data_format;
    DataType blob_datatype     = blob_->GetBlobDesc().data_type;
    LOGD("Convert From Mat:  mat type: %d  mat device type: %d\n", mat.GetMatType(), mat.GetDeviceType());
    if (NCHW_FLOAT == mat.GetMatType()) {
        if (DATA_FORMAT_NCHW == blob_dataformat && DATA_TYPE_FLOAT == blob_datatype) {
            tnn_ret = AtlasMemoryCopyAsync(blob_->GetHandle().base, mat.GetData(), mat.GetDeviceType(), mat_bytesize,
                                           atlas_cmd_queue->stream, true);
            if (tnn_ret != TNN_OK)
                return tnn_ret;
        } else if (DATA_FORMAT_NHWC == blob_dataformat && DATA_TYPE_FLOAT == blob_datatype) {
            // only support DEVICE_NAIVE device type
            if (DEVICE_NAIVE == mat.GetDeviceType()) {
                if (nullptr == buffer_) {
                    buffer_.reset(new char[mat_bytesize], [](char *p) { delete[] p; });
                }
                // transfer from NCHW to NHWC
                LOGD("convert from nchw to nhwc\n");
                auto blob_dim = blob_->GetBlobDesc().dims;
                DataFormatConverter::ConvertFromNCHWToNHWCFloat((float *)mat.GetData(), (float *)buffer_.get(),
                                                                blob_dim[0], blob_dim[3], blob_dim[1], blob_dim[2]);

                tnn_ret = AtlasMemoryCopyAsync(blob_->GetHandle().base, buffer_.get(), DEVICE_NAIVE, mat_bytesize,
                                               atlas_cmd_queue->stream, true);
                if (tnn_ret != TNN_OK)
                    return tnn_ret;
            } else {
                return Status(TNNERR_PARAM_ERR, "not support this device type convert in no-aipp model yet!");
            }
        } else {
            return Status(TNNERR_PARAM_ERR, "not support this dataformat type convert in no-aipp model yet!");
        }
    } else if (NC_INT32 == mat.GetMatType()) {
        LOGD("Convert from NC_INT32 Mat: mat device type: %d\n", mat.GetDeviceType());
        if (DATA_TYPE_INT32 == blob_datatype) {
            tnn_ret = AtlasMemoryCopyAsync(blob_->GetHandle().base, mat.GetData(), mat.GetDeviceType(), mat_bytesize,
                                           atlas_cmd_queue->stream, true);
            if (tnn_ret != TNN_OK)
                return tnn_ret;
        } else if (DATA_TYPE_FLOAT == blob_datatype) {
            LOGD("WARNING: Target Blob name is '%s', internally convert Blob DataType from FLOAT to INT32.", blob_->GetBlobDesc().name.c_str());
            blob_->GetBlobDesc().data_type = DATA_TYPE_INT32;
            tnn_ret = AtlasMemoryCopyAsync(blob_->GetHandle().base, mat.GetData(), mat.GetDeviceType(), mat_bytesize,
                                           atlas_cmd_queue->stream, true);
            if (tnn_ret != TNN_OK)
                return tnn_ret;
        } else {
            LOGE("Convert From NC_INT32 Mat: target blob Should be INT32, or FLOAT(convert to INT32)but got %d: %d\n", int(blob_datatype));
            return Status(TNNERR_PARAM_ERR, "Convert From NC_INT32 Mat: target blob Should be INT32.");
        }
    } else if (NC_INT64 == mat.GetMatType()) {
        LOGD("Convert from NC_INT64 Mat: mat device type: %d\n", mat.GetDeviceType());
        if (DATA_TYPE_INT64 == blob_datatype) {
            tnn_ret = AtlasMemoryCopyAsync(blob_->GetHandle().base, mat.GetData(), mat.GetDeviceType(), mat_bytesize,
                                           atlas_cmd_queue->stream, true);
            if (tnn_ret != TNN_OK)
                return tnn_ret;
        } else if (DATA_TYPE_FLOAT == blob_datatype || DATA_TYPE_INT32 == blob_datatype) {
            LOGD("WARNING: Target Blob name is '%s', internally convert Blob DataType from FLOAT/INT32 to INT64.", blob_->GetBlobDesc().name.c_str());
            blob_->GetBlobDesc().data_type   = DATA_TYPE_INT64;
            BlobMemorySizeInfo new_size_info = Calculate1DMemorySize(blob_->GetBlobDesc());
            blob_bytesize_                   = GetBlobMemoryBytesSize(new_size_info);  // sizeof(int64_t) == 8, re-calculate ByteSize
            tnn_ret = AtlasMemoryCopyAsync(blob_->GetHandle().base, mat.GetData(), mat.GetDeviceType(), mat_bytesize,
                                           atlas_cmd_queue->stream, true);
            if (tnn_ret != TNN_OK)
                return tnn_ret;
        } else {
            LOGE("Convert From NC_INT64 Mat: target blob Should be INT64, or FLOAT(convert to INT64)but got %d: %d\n", int(blob_datatype));
            return Status(TNNERR_PARAM_ERR, "Convert From NC_INT64 Mat: target blob Should be INT64.");
        }
    } else {
        return Status(TNNERR_PARAM_ERR, "not support this mat type convert in no-aipp model yet!");
    }

    return TNN_OK;
}

Status AtlasBlobConverterAcc::ConvertFromMatAsyncWithStaticAipp(Mat &mat, MatConvertParam param,
                                                                AtlasCommandQueue *atlas_cmd_queue) {
    Status tnn_ret   = TNN_OK;
    aclError acl_ret = ACL_ERROR_NONE;

    do_scale_bias_ = NeedDoScaleBias(param);

    if (do_scale_bias_) {
        LOGE("warning: mat convert param is useless in no-dynamic aipp model!\n");
    }

    int mat_bytesize = 0;
    tnn_ret          = MatUtils::GetMatByteSize(mat, mat_bytesize);
    if (TNN_OK != tnn_ret) {
        LOGE("GetMatByteSize failed in ConvertFromMatAsyncWithoutAipp\n");
        return tnn_ret;
    }

    LOGD("Convert From Mat:  mat type: %d  mat device type: %d  acl input format:%d\n", mat.GetMatType(),
         mat.GetDeviceType(), model_info_.aipp_input_format);
    if ((N8UC3 == mat.GetMatType() && ACL_RGB888_U8 == model_info_.aipp_input_format) ||
        (NGRAY == mat.GetMatType() && ACL_YUV400_U8 == model_info_.aipp_input_format) ||
        ((NNV12 == mat.GetMatType() || NNV21 == mat.GetMatType()) &&
         ACL_YUV420SP_U8 == model_info_.aipp_input_format)) {
        tnn_ret = AtlasMemoryCopyAsync(blob_->GetHandle().base, mat.GetData(), mat.GetDeviceType(), mat_bytesize,
                                       atlas_cmd_queue->stream, true);
        if (tnn_ret != TNN_OK)
            return tnn_ret;
    } else {
        return Status(TNNERR_PARAM_ERR, "input mat type mismatch with static aipp!");
    }

    return TNN_OK;
}

Status AtlasBlobConverterAcc::ConvertFromMatAsyncWithDynamicAipp(Mat &mat, MatConvertParam param,
                                                                 AtlasCommandQueue *atlas_cmd_queue) {
    Status tnn_ret = SetDynamicAipp(mat, param);
    if (TNN_OK != tnn_ret) {
        LOGE("set dynamic aipp failed!\n");
        return tnn_ret;
    }
    auto data_buffer = aclmdlGetDatasetBuffer(model_info_.input_dataset, 0);
    if (nullptr == data_buffer) {
        LOGE("get data buffer from dataset failed!\n");
        return Status(TNNERR_ATLAS_RUNTIME_ERROR, "get data buffer failed");
    }

    auto data_buffer_ptr = aclGetDataBufferAddr(data_buffer);
    if (nullptr == data_buffer_ptr) {
        LOGE("get data buffer from dataset failed!\n");
        return Status(TNNERR_ATLAS_RUNTIME_ERROR, "get data buffer failed");
    }

    if (blob_->GetHandle().base != data_buffer_ptr) {
        LOGE("data buffer ptr not match blob data ptr (0x%lx vs 0x%lx)! note: dynamic aipp not support multi input\n",
             (unsigned long)data_buffer_ptr, (unsigned long)blob_->GetHandle().base);
        return Status(TNNERR_ATLAS_RUNTIME_ERROR, "data buffer ptr is invalid");
    }

    int mat_bytesize = 0;
    tnn_ret          = MatUtils::GetMatByteSize(mat, mat_bytesize);
    if (TNN_OK != tnn_ret) {
        LOGE("GetMatByteSize failed in ConvertFromMatAsyncWithDynamicAipp\n");
        return tnn_ret;
    }

    tnn_ret = AtlasMemoryCopyAsync(data_buffer_ptr, mat.GetData(), mat.GetDeviceType(), mat_bytesize,
                                   atlas_cmd_queue->stream, true);

    return tnn_ret;
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

Status AtlasBlobConverterAcc::AtlasMemoryCopyAsync(void *dst, void *src, DeviceType mat_device_type, int bytes,
                                                   void *stream, bool from_mat) {
    aclError ret = ACL_ERROR_NONE;
    if (DEVICE_ATLAS == mat_device_type) {
        // need to copy from device to device
        LOGD("acl memcpy: copy from device to device (%d bytes)\n", bytes);
        ret = aclrtMemcpyAsync(dst, bytes, src, bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
        if (ACL_ERROR_NONE != ret) {
            return Status(TNNERR_ATLAS_RUNTIME_ERROR, "acl memory copy failed");
        }
    } else if (DEVICE_NAIVE == mat_device_type || DEVICE_ARM == mat_device_type) {
        if (from_mat) {
            // need to copy from host to device
            LOGD("acl memcpy: copy from host to device (%d bytes)\n", bytes);
            ret = aclrtMemcpyAsync(dst, bytes, src, bytes, ACL_MEMCPY_HOST_TO_DEVICE, stream);
        } else {
            // need to copy form device to host
            LOGD("acl memcpy: copy from device to host (%d bytes)\n", bytes);
            ret = aclrtMemcpyAsync(dst, bytes, src, bytes, ACL_MEMCPY_DEVICE_TO_HOST, stream);
        }
        if (ACL_ERROR_NONE != ret) {
            return Status(TNNERR_ATLAS_RUNTIME_ERROR, "acl memory copy failed");
        }
    } else {
        return Status(TNNERR_PARAM_ERR, "not support this device type convert yet!");
    }

    return TNN_OK;
}

Status AtlasBlobConverterAcc::SetDynamicAipp(Mat &mat, MatConvertParam &param) {
    aclError acl_ret = ACL_ERROR_NONE;
    Status tnn_ret   = TNN_OK;

    if (nullptr == aipp_dynamic_set_) {
        aipp_mat_batchsize_ = GetMaxBatchSize(model_info_.model_desc, blob_->GetBlobDesc().dims[0]);
        aipp_dynamic_set_   = aclmdlCreateAIPP(aipp_mat_batchsize_);
        if (nullptr == aipp_dynamic_set_) {
            LOGE("create aipp info failed\n");
            return Status(TNNERR_ATLAS_RUNTIME_ERROR, "create aipp info failed!\n");
        }
    }

    int height = mat.GetHeight();
    int width  = mat.GetWidth();

    // set aipp image size
    acl_ret = aclmdlSetAIPPSrcImageSize(aipp_dynamic_set_, width, height);
    if (ACL_ERROR_NONE != acl_ret) {
        return Status(TNNERR_ATLAS_RUNTIME_ERROR, "aipp set image size failed!\n");
    }
    LOGD("set aipp input image size: w = %d  h = %d\n", width, height);

    // set aipp input format
    aclAippInputFormat aipp_input_format;
    tnn_ret = ConvertFromMatTypeToAippInputFormat(mat.GetMatType(), aipp_input_format);
    if (TNN_OK != tnn_ret) {
        return tnn_ret;
    }
    acl_ret = aclmdlSetAIPPInputFormat(aipp_dynamic_set_, aipp_input_format);
    if (ACL_ERROR_NONE != acl_ret) {
        return Status(TNNERR_ATLAS_RUNTIME_ERROR, "aipp set image format failed!\n");
    }
    LOGD("set aipp input format: %d\n", aipp_input_format);

    // set aipp mean and var
    float aipp_mean0 = (-1.0f) * param.bias[0] / param.scale[0];
    float aipp_mean1 = (-1.0f) * param.bias[1] / param.scale[1];
    float aipp_mean2 = (-1.0f) * param.bias[2] / param.scale[2];
    float aipp_mean3 = (-1.0f) * param.bias[3] / param.scale[3];
    for (int i = 0; i < mat.GetBatch(); ++i) {
        acl_ret = aclmdlSetAIPPDtcPixelMin(aipp_dynamic_set_, aipp_mean0, aipp_mean1, aipp_mean2, aipp_mean3, i);
        if (ACL_ERROR_NONE != acl_ret) {
            return Status(TNNERR_ATLAS_RUNTIME_ERROR, "aipp set mean failed!\n");
        }
        LOGD("set aipp input mean: %f, %f, %f, %f\n", aipp_mean0, aipp_mean1, aipp_mean2, aipp_mean3);
        acl_ret = aclmdlSetAIPPPixelVarReci(aipp_dynamic_set_, param.scale[0], param.scale[1], param.scale[2],
                                            param.scale[3], i);
        if (ACL_ERROR_NONE != acl_ret) {
            return Status(TNNERR_ATLAS_RUNTIME_ERROR, "aipp set var failed!\n");
        }
        LOGD("set aipp input scale: %f, %f, %f, %f\n", param.scale[0], param.scale[1], param.scale[2], param.scale[3]);
    }

    // set aipp ax swap
    if (ACL_XRGB8888_U8 == aipp_input_format) {
        acl_ret = aclmdlSetAIPPAxSwapSwitch(aipp_dynamic_set_, 1);
        if (ACL_ERROR_NONE != acl_ret) {
            return Status(TNNERR_ATLAS_RUNTIME_ERROR, "aipp set ax swap failed!\n");
        }
        LOGD("set aipp ax swap switch: 1\n");
    }

    // convert format
    {
        // if input is yuv, then use csc to convert from yuv to rgb
        if (ACL_YUV420SP_U8 == aipp_input_format) {
            acl_ret = aclmdlSetAIPPCscParams(aipp_dynamic_set_, 1, 256, 0, 359, 256, -88, -183, 256, 454, 0, 0, 0, 0, 0,
                                             128, 128);
            if (ACL_ERROR_NONE != acl_ret) {
                return Status(TNNERR_ATLAS_RUNTIME_ERROR, "aipp set csc failed!\n");
            }
            LOGD("set aipp csc params\n");
        }

        // set aipp swap
        if (ACL_RGB888_U8 == aipp_input_format || ACL_XRGB8888_U8 == aipp_input_format) {
            acl_ret = aclmdlSetAIPPRbuvSwapSwitch(aipp_dynamic_set_, (int8_t)param.reverse_channel);
            LOGD("set aipp rbuv swap switch: %d\n", param.reverse_channel);
        } else if (ACL_YUV420SP_U8 == aipp_input_format) {
            if (NNV12 == mat.GetMatType()) {
                acl_ret = aclmdlSetAIPPRbuvSwapSwitch(aipp_dynamic_set_, (int8_t)param.reverse_channel);
                LOGD("set aipp rbuv swap switch: %d\n", param.reverse_channel);
            } else if (NNV21 == mat.GetMatType()) {
                // opposite with param.reverse_channel
                if (param.reverse_channel) {
                    acl_ret = aclmdlSetAIPPRbuvSwapSwitch(aipp_dynamic_set_, 0);
                    LOGD("set aipp rbuv swap switch: %d\n", 0);
                } else {
                    acl_ret = aclmdlSetAIPPRbuvSwapSwitch(aipp_dynamic_set_, 1);
                    LOGD("set aipp rbuv swap switch: %d\n", 1);
                }
            }
        }

        if (ACL_ERROR_NONE != acl_ret) {
            return Status(TNNERR_ATLAS_RUNTIME_ERROR, "aipp set swap failed!\n");
        }
    }

    // set input aipp
    acl_ret =
        aclmdlSetInputAIPP(model_info_.model_id, model_info_.input_dataset, dynamic_aipp_index_, aipp_dynamic_set_);
    if (ACL_ERROR_NONE != acl_ret) {
        return Status(TNNERR_ATLAS_RUNTIME_ERROR, "aipp set input failed!\n");
    }

    return TNN_OK;
}

DECLARE_BLOB_CONVERTER_CREATER(Atlas);
REGISTER_BLOB_CONVERTER(Atlas, DEVICE_ATLAS);

}  // namespace TNN_NS
