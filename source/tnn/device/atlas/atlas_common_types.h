// Copyright 2019 Tencent. All Rights Reserved

#ifndef TNN_SOURCE_DEVICE_ATLAS_ATLAS_COMMON_TYPES_H_
#define TNN_SOURCE_DEVICE_ATLAS_ATLAS_COMMON_TYPES_H_

#include <map>
#include <memory>
#include <string>
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
    std::string om_path;
    uint32_t graph_id  = 0;
    bool with_dvpp     = false;
    bool dynamic_aipp  = false;
    bool daipp_swap_rb = false;
    bool daipp_norm    = false;
    int height;
    int width;
    int dvpp_engine_id      = 252;
    int inference_engine_id = 535;
    int output_engine_id    = 480;
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

}  // namespace TNN_NS

#endif  // TNN_SOURCE_DEVICE_ATLAS_ATLAS_COMMON_TYPES_H_
