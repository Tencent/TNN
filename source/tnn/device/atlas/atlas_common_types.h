// Copyright 2019 Tencent. All Rights Reserved

#ifndef TNN_SOURCE_DEVICE_ATLAS_ATLAS_COMMON_TYPES_H_
#define TNN_SOURCE_DEVICE_ATLAS_ATLAS_COMMON_TYPES_H_

#include <map>
#include <memory>
#include <string>
#include "acl/acl.h"
#include "tnn/core/blob.h"
#include "tnn/core/macro.h"

namespace TNN_NS {

enum ImageTypeT {
    IMAGE_TYPE_RAW  = -1,
    IMAGE_TYPE_NV12 = 0,
    IMAGE_TYPE_JPEG,
    IMAGE_TYPE_PNG,
    IMAGE_TYPE_BMP,
    IMAGE_TYPE_TIFF,
    IMAGE_TYPE_VIDEO = 100
};

struct AtlasModelConfig {
    std::string om_str = "";
    bool is_path       = false;
};

struct DimInfo {
    uint32_t batch   = 0;
    uint32_t channel = 0;
    uint32_t height  = 0;
    uint32_t width   = 0;
};

struct AtlasCommandQueue {
    void* context;
    void* stream;
};

struct AtlasModelInfo {
    aclmdlDesc* model_desc               = nullptr;
    uint32_t model_id                    = 0;
    aclmdlDataset* input_dataset         = nullptr;
    bool has_aipp                        = false;
    aclAippInputFormat aipp_input_format = ACL_AIPP_RESERVED;
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_DEVICE_ATLAS_ATLAS_COMMON_TYPES_H_
