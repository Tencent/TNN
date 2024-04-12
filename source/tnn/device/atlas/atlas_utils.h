// Copyright 2019 Tencent. All Rights Reserved

#ifndef TNN_SOURCE_DEVICE_ATLAS_ATLAS_UTILS_H_
#define TNN_SOURCE_DEVICE_ATLAS_ATLAS_UTILS_H_

#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>
#include "acl/acl.h"
#include "acl/ops/acl_dvpp.h"
#include "tnn/core/common.h"
#include "tnn/core/macro.h"
#include "tnn/core/status.h"
#include "tnn/utils/blob_converter.h"

namespace TNN_NS {

Status ConvertFromAclDataTypeToTnnDataType(aclDataType acl_datatype, DataType& tnn_datatype);

Status ConvertAclDataFormatToTnnDataFormat(aclFormat acl_format, DataFormat& tnn_dataformat);

Status ConvertFromMatTypeToAippInputFormat(MatType mat_type, aclAippInputFormat& aipp_input_format);

Status ConvertFromMatTypeToDvppPixelFormat(MatType mat_type, acldvppPixelFormat& dvpp_pixel_format);

bool IsDynamicBatch(aclmdlDesc* model_desc, std::string input_name);

int GetMaxBatchSize(aclmdlDesc *desc, int default_batch);

}  // namespace TNN_NS

#endif  // TNN_SOURCE_DEVICE_ATLAS_ATLAS_UTILS_H_
