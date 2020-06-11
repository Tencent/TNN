// Copyright 2019 Tencent. All Rights Reserved

#ifndef TNN_SOURCE_DEVICE_ATLAS_ENGINES_INFERENCE_ENGINE_DVPP_ENGINE_H_
#define TNN_SOURCE_DEVICE_ATLAS_ENGINES_INFERENCE_ENGINE_DVPP_ENGINE_H_

#include <utility>
#include "atlas_common_types.h"
#include "dvpp/Vpc.h"
#include "dvpp/idvppapi.h"
#include "hiaiengine/ai_model_manager.h"
#include "hiaiengine/engine.h"
#include "tnn/core/macro.h"

#define DT_INPUT_SIZE 1
#define DT_OUTPUT_SIZE 1

namespace TNN_NS {

struct DvppConfig {
    int point_x;      // the abscissa of crop(must be even)
    int point_y;      // the ordinate of crop(must be even)
    int crop_width;   // the width of crop(must be even)
    int crop_height;  // the height of crop(must be even)
    int self_crop;    // user defined
    int dump_value;   // the switch of dump image after preProcess, 0:no dump,
                      // 1:dump
    std::string project_name;
    float resize_width;
    float resize_height;
    std::string userHome;

    bool crop_before_resize;  // whether resize is before crop

    bool yuv420_need;    // whether to need yuv420 semi-planar to output data
                         // Jpegd support raw and yuv420sp output, while op_type
                         // is set to 1
    bool v_before_u;     // output sp format, uv order. whether v is before u,
                         // while op_type is set to 1
    int transform_flag;  // format transform flag, while op_type is set to 2
    std::string dvpp_para;  // dvpp para path file name

    DvppConfig() {
        point_x            = -1;
        point_y            = -1;
        crop_width         = -1;
        crop_height        = -1;
        self_crop          = 1;
        dump_value         = 0;
        project_name       = "";
        resize_width       = 0;
        resize_height      = 0;
        crop_before_resize = true;
        yuv420_need        = false;
        v_before_u         = true;
        transform_flag     = 0;
        dvpp_para          = "";
        userHome           = "";
    }
};

// self_defined, reason as follows by dvpp, png format pic only supports RGB &
// RGBA
enum pngd_color_space {
    DVPP_PNG_DECODE_OUT_RGB  = 2,
    DVPP_PNG_DECODE_OUT_RGBA = 6
};

enum FILE_TYPE {
    FILE_TYPE_PIC_JPEG = 0x1,
    FILE_TYPE_PIC_PNG,
    FILE_TYPE_YUV,
    FILE_TYPE_MAX
};

class DvppEngine : public hiai::Engine {
public:
    DvppEngine();
    ~DvppEngine();

    HIAI_StatusT Init(const hiai::AIConfig &config,
                      const std::vector<hiai::AIModelDescription> &model_desc);

    HIAI_DEFINE_PROCESS(DT_INPUT_SIZE, DT_OUTPUT_SIZE)

private:
    void ClearData();
    bool NeedCrop();
    bool ProcessCrop(VpcUserCropConfigure &area, const int &width,
                     const int &height, const int &realWidth,
                     const int &realHeight);

    int HandleDvpp();
    int HandleJpeg(const hiai::ImageData<uint8_t> &img);
    int HandlePng(const hiai::ImageData<uint8_t> &img);
    int HandleVpc(const hiai::ImageData<uint8_t> &img);
    int HandleVpcWithParam(const unsigned char *buffer, const int &width,
                           const int &height, const long &bufferSize,
                           const hiai::ImageData<uint8_t> &img,
                           const FILE_TYPE &type, const int &format);

    bool SendPreProcessData();

private:
    std::shared_ptr<DvppConfig> dvpp_config_;
    std::shared_ptr<DvppInputDataType> dvpp_out_;
    std::shared_ptr<DvppInputDataType> dvpp_in_;
    IDVPPAPI *pidvppapi_;
    uint32_t image_frame_id_;
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_DEVICE_ATLAS_ENGINES_INFERENCE_ENGINE_DVPP_ENGINE_H_
