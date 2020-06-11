// Copyright 2019 Tencent. All Rights Reserved

#ifndef TNN_SOURCE_DEVICE_ATLAS_ATLAS_COMMON_TYPES_H_
#define TNN_SOURCE_DEVICE_ATLAS_ATLAS_COMMON_TYPES_H_

#include <cereal/types/map.hpp>
#include <cereal/types/vector.hpp>
#include <map>
#include <memory>
#include <string>
#include "tnn/core/macro.h"
#include "hiaiengine/data_type.h"
#include "hiaiengine/data_type_reg.h"

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

enum CommandType {
    CT_None             = 0,
    CT_DataTransfer     = 1,
    CT_DataTransfer_End = 2,
    CT_InfoQuery        = 3,
    CT_InfoQuery_End    = 4,
};

enum QueryType { QT_None = 0, QT_InputDims = 1, QT_OutputDims = 2 };

struct DimInfo {
    uint32_t batch   = 0;
    uint32_t channel = 0;
    uint32_t height  = 0;
    uint32_t width   = 0;
};
template <class Archive>
void serialize(Archive& ar, DimInfo& data);

struct TransferDataInfo {
    CommandType cmd_type = CT_None;
    QueryType query_type = QT_None;
    DimInfo dim_info;
    uint32_t size_in_bytes = 0;
    char name[32]          = "";
};
template <class Archive>
void serialize(Archive& ar, TransferDataInfo& data);

struct OutputDataInfo {
    std::map<std::string, long> output_map;
    long output_cv_addr        = 0;
    long time_s                = 0;
    long time_ns               = 0;
    double process_duration_ms = 0;
};
template <class Archive>
void serialize(Archive& ar, OutputDataInfo& data);

struct TransferDataType {
    TransferDataInfo info;
    OutputDataInfo output_info;
    uint32_t data_len = 0;
    std::shared_ptr<uint8_t> data;
};
template <class Archive>
void serialize(Archive& ar, TransferDataType& data);

struct DvppInputDataType {
    OutputDataInfo output_info;
    hiai::BatchInfo b_info;
    std::vector<hiai::ImageData<uint8_t>> img_vec;
};

struct DvppTransDataType {
    OutputDataInfo output_info;
    hiai::BatchInfo b_info;
    hiai::ImageData<uint8_t> img_data;
};
template <class Archive>
void serialize(Archive& ar, DvppTransDataType& data);

void GetTransferDataTypeSearPtr(void* input_ptr, std::string& ctrl_str,
                                uint8_t*& data_ptr, uint32_t& data_len);

std::shared_ptr<void> GetTransferDataTypeDearPtr(const char* ctrl_ptr,
                                                 const uint32_t& ctr_len,
                                                 const uint8_t* data_ptr,
                                                 const uint32_t& data_len);

}  // namespace TNN_NS

#endif  // TNN_SOURCE_DEVICE_ATLAS_ATLAS_COMMON_TYPES_H_
