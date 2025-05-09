// Copyright 2019 Tencent. All Rights Reserved

#include "atlas_utils.h"
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>

namespace TNN_NS {

Status ConvertFromAclDataTypeToTnnDataType(aclDataType acl_datatype, DataType& tnn_datatype) {
    if (ACL_FLOAT == acl_datatype) {
        tnn_datatype = DATA_TYPE_FLOAT;
    } else if (ACL_FLOAT16 == acl_datatype) {
        tnn_datatype = DATA_TYPE_HALF;
    } else if (ACL_INT8 == acl_datatype || ACL_UINT8 == acl_datatype) {
        tnn_datatype = DATA_TYPE_INT8;
    } else if (ACL_INT32 == acl_datatype || ACL_UINT32 == acl_datatype) {
        tnn_datatype = DATA_TYPE_INT32;
    } else if (ACL_INT64 == acl_datatype || ACL_UINT64 == acl_datatype) {
        tnn_datatype = DATA_TYPE_INT64;
    } else {
        LOGE("not support convert from acl datatype (%d) to tnn datatype\n", acl_datatype);
        return Status(TNNERR_COMMON_ERROR, "the data type is not support");
    }
    return TNN_OK;
}

Status ConvertAclDataFormatToTnnDataFormat(aclFormat acl_format, DataFormat& tnn_dataformat) {
    if (ACL_FORMAT_NCHW == acl_format || ACL_FORMAT_ND == acl_format) {
        tnn_dataformat = DATA_FORMAT_NCHW;
    } else if (ACL_FORMAT_NHWC == acl_format) {
        tnn_dataformat = DATA_FORMAT_NHWC;
    } else {
        LOGE("not support convert from acl dataformat (%d) to tnn datatype\n", acl_format);
        return Status(TNNERR_COMMON_ERROR, "the data format is not support");
    }
    return TNN_OK;
}

Status ConvertFromMatTypeToAippInputFormat(MatType mat_type, aclAippInputFormat& aipp_input_format) {
    if (N8UC3 == mat_type) {
        aipp_input_format = ACL_RGB888_U8;
    } else if (N8UC4 == mat_type) {
        aipp_input_format = ACL_XRGB8888_U8;
    } else if (NNV12 == mat_type || NNV21 == mat_type) {
        aipp_input_format = ACL_YUV420SP_U8;
    } else if (NGRAY == mat_type) {
        aipp_input_format = ACL_YUV400_U8;
    } else {
        LOGE("not support convert from mat type (%d) to aipp input format\n", mat_type);
        return Status(TNNERR_ATLAS_AIPP_NOT_SUPPORT, "the mat type is not support");
    }

    return TNN_OK;
}

Status ConvertFromMatTypeToDvppPixelFormat(MatType mat_type, acldvppPixelFormat& dvpp_pixel_format) {
    if (N8UC3 == mat_type) {
        dvpp_pixel_format = PIXEL_FORMAT_RGB_888;
    } else if (N8UC4 == mat_type) {
        dvpp_pixel_format = PIXEL_FORMAT_RGBA_8888;
    } else if (NNV12 == mat_type) {
        dvpp_pixel_format = PIXEL_FORMAT_YUV_SEMIPLANAR_420;
    } else if (NNV21 == mat_type) {
        dvpp_pixel_format = PIXEL_FORMAT_YVU_SEMIPLANAR_420;
    } else {
        LOGE("not support convert from mat type (%d) to dvpp pixel format\n", mat_type);
        return Status(TNNERR_ATLAS_DVPP_NOT_SUPPORT, "the mat type is not support");
    }

    return TNN_OK;
}

bool IsDynamicBatch(aclmdlDesc* model_desc, std::string input_name) {
    size_t index     = 0;
    aclError acl_ret = aclmdlGetInputIndexByName(model_desc, input_name.c_str(), &index);
    if (ACL_ERROR_NONE != acl_ret) {
        return false;
    }

    aclmdlIODims acl_dims;
    acl_ret = aclmdlGetInputDims(model_desc, index, &acl_dims);
    if (ACL_ERROR_NONE != acl_ret) {
        return false;
    }

    if (-1 == acl_dims.dims[0]) {
        return true;
    }
    return false;
}

int GetMaxBatchSize(aclmdlDesc *desc, int default_batch) {
    aclmdlBatch batch_info;

    aclError acl_ret = aclmdlGetDynamicBatch(desc, &batch_info);
    if (ACL_ERROR_NONE != acl_ret) {
        LOGE("get dynamic batch info failed\n");
        return 0;
    }

    int max_batchsize = 0;
    if (batch_info.batchCount > 0) {
        // dynamic batch
        for (int i = 0; i < batch_info.batchCount; ++i) {
            if (batch_info.batch[i] > max_batchsize) {
                max_batchsize = batch_info.batch[i];
            }
        }
    } else {
        // static batch
        max_batchsize = default_batch;
    }

    LOGD("get max batch size: %d\n", max_batchsize);
    return max_batchsize;
}

}  // namespace TNN_NS
