// Copyright 2019 Tencent. All Rights Reserved

#ifndef TNN_SOURCE_DEVICE_ATLAS_ATLAS_UTILS_H_
#define TNN_SOURCE_DEVICE_ATLAS_ATLAS_UTILS_H_

#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>
#include "acl/acl.h"
#include "tnn/core/common.h"
#include "tnn/core/macro.h"
#include "tnn/core/status.h"

namespace TNN_NS {

std::vector<std::string> SplitPath(const std::string& str, const std::set<char> delimiters);

long GetCurentTime();

int SaveMemToFile(std::string file_name, void* data, int size);

Status ConvertFromAclDataTypeToTnnDataType(aclDataType acl_datatype, DataType& tnn_datatype);

Status ConvertFromAclDataFormatToTnnDataFormat(aclFormat acl_format, DataFormat& tnn_dataformat);

}  // namespace TNN_NS

#endif  // TNN_SOURCE_DEVICE_ATLAS_ATLAS_UTILS_H_
