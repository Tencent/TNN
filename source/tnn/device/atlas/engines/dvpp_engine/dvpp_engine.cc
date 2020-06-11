// Copyright 2019 Tencent. All Rights Reserved

#include "dvpp_engine.h"
#include <time.h>
#include <memory>
#include "atlas_device_utils.h"
#include "atlas_utils.h"
#include "hiaiengine/ai_memory.h"
#include "hiaiengine/c_graph.h"
#include "hiaiengine/data_type.h"
#include "hiaiengine/log.h"

namespace TNN_NS {

const static int DVPP_SUPPORT_MAX_WIDTH  = 4096;
const static int DVPP_SUPPORT_MIN_WIDTH  = 16;
const static int DVPP_SUPPORT_MAX_HEIGHT = 4096;
const static int DVPP_SUPPORT_MIN_HEIGHT = 16;

#define CHECK_ODD(NUM) (((NUM) % 2 != 0) ? (NUM) : ((NUM)-1))
#define CHECK_EVEN(NUM) (((NUM) % 2 == 0) ? (NUM) : ((NUM)-1))

DvppEngine::DvppEngine() {
    dvpp_config_ = nullptr;
    dvpp_out_    = nullptr;
    dvpp_in_     = nullptr;
    pidvppapi_   = nullptr;
}

DvppEngine::~DvppEngine() {
    if (nullptr != pidvppapi_) {
        DestroyDvppApi(pidvppapi_);
        pidvppapi_ = nullptr;
    }
    HIAI_ENGINE_LOG(HIAI_IDE_INFO, "[DvppEngine]DvppEngine Engine Destory");
}

HIAI_StatusT DvppEngine::Init(
    const hiai::AIConfig &config,
    const std::vector<hiai::AIModelDescription> &modelDesc) {
    HIAI_ENGINE_LOG(HIAI_IDE_INFO, "[DvppEngine] start init!\n");

    if (nullptr == dvpp_config_) {
        dvpp_config_ = std::make_shared<DvppConfig>();
        if (nullptr == dvpp_config_ || nullptr == dvpp_config_.get()) {
            HIAI_ENGINE_LOG(
                HIAI_IDE_ERROR,
                "[DvppEngine] call dvpp_config_ make_shared failed");
            return HIAI_ERROR;
        }
    }

    // get config from DvppEngine Property setting of user.
    std::stringstream ss;
    for (int index = 0; index < config.items_size(); ++index) {
        const ::hiai::AIConfigItem &item = config.items(index);
        std::string name                 = item.name();
        ss << item.value();
        if ("point_x" == name) {
            ss >> (*dvpp_config_).point_x;
        } else if ("point_y" == name) {
            ss >> (*dvpp_config_).point_y;
        } else if ("crop_width" == name) {
            ss >> (*dvpp_config_).crop_width;
        } else if ("crop_height" == name) {
            ss >> (*dvpp_config_).crop_height;
        } else if ("self_crop" == name) {
            ss >> (*dvpp_config_).self_crop;
        } else if ("dump_value" == name) {
            ss >> (*dvpp_config_).dump_value;
        } else if ("project_name" == name) {
            ss >> (*dvpp_config_).project_name;
        } else if ("resize_height" == name) {
            ss >> (*dvpp_config_).resize_height;
        } else if ("resize_width" == name) {
            ss >> (*dvpp_config_).resize_width;
        } else if ("crop_before_resize" == name) {
            ss >> (*dvpp_config_).crop_before_resize;
        } else if ("yuv420_need" == name) {
            ss >> (*dvpp_config_).yuv420_need;
        } else if ("v_before_u" == name) {
            ss >> (*dvpp_config_).v_before_u;
        } else if ("transform_flag" == name) {
            ss >> (*dvpp_config_).transform_flag;
        } else if ("dvpp_parapath" == name) {
            ss >> (*dvpp_config_).dvpp_para;
        } else if ("userHome" == name) {
            ss >> (*dvpp_config_).userHome;
        } else {
            std::string userDefined = "";
            ss >> userDefined;
            HIAI_ENGINE_LOG(HIAI_IDE_INFO, "userDefined:name[%s], value[%s]",
                            name.c_str(), userDefined.c_str());
        }
        ss.clear();
    }

    // check crop param
    if (NeedCrop()) {
        if (dvpp_config_->point_x >
                DVPP_SUPPORT_MAX_WIDTH - DVPP_SUPPORT_MIN_WIDTH - 1 ||
            dvpp_config_->crop_width > DVPP_SUPPORT_MAX_WIDTH ||
            dvpp_config_->crop_width < DVPP_SUPPORT_MIN_WIDTH ||
            dvpp_config_->point_y >
                DVPP_SUPPORT_MAX_HEIGHT - DVPP_SUPPORT_MIN_HEIGHT - 1 ||
            dvpp_config_->crop_height > DVPP_SUPPORT_MAX_HEIGHT ||
            dvpp_config_->crop_height < DVPP_SUPPORT_MIN_HEIGHT) {
            HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "crop param error");
            return HIAI_ERROR;
        }
    }

    if (DVPP_SUCCESS != CreateDvppApi(pidvppapi_)) {
        HIAI_ENGINE_LOG(HIAI_IDE_ERROR,
                        "[DvppEngine]Create DVPP pidvppapi_ fail");
        return HIAI_ERROR;
    }

    HIAI_ENGINE_LOG(HIAI_IDE_INFO, "[DvppEngine] end init!\n");
    return HIAI_OK;
}

HIAI_IMPL_ENGINE_PROCESS("DvppEngine", DvppEngine, DT_INPUT_SIZE) {
    HIAI_ENGINE_LOG(HIAI_IDE_INFO, "[DvppEngine] process start\n");
    HIAI_StatusT ret = HIAI_OK;
    std::shared_ptr<DvppTransDataType> input_arg =
        std::static_pointer_cast<DvppTransDataType>(arg0);
    if (nullptr == input_arg) {
        HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "Fail to process invalid message\n");
        return HIAI_ERROR;
    }

    if (nullptr == dvpp_in_) {
        dvpp_in_ = std::make_shared<DvppInputDataType>();
        if (dvpp_in_ == nullptr) {
            HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "[DvppEngine] malloc error");
            return HIAI_ERROR;
        }

        dvpp_in_->output_info = input_arg->output_info;
        dvpp_in_->b_info      = input_arg->b_info;
        dvpp_in_->img_vec.push_back(input_arg->img_data);
    } else {
        if (dvpp_in_->b_info.batch_ID == input_arg->b_info.batch_ID &&
            !input_arg->b_info.frame_ID.empty()) {
            dvpp_in_->output_info = input_arg->output_info;  // to overlap
            dvpp_in_->img_vec.push_back(input_arg->img_data);
            dvpp_in_->b_info.frame_ID.push_back(input_arg->b_info.frame_ID[0]);
        }
    }
    HIAI_ENGINE_LOG(HIAI_IDE_INFO,
                    "[DvppEngine] Input packet info: batch_id:%d  frame_id:%d  "
                    "input size: %u\n",
                    input_arg->b_info.batch_ID, input_arg->b_info.frame_ID[0],
                    dvpp_in_->b_info.frame_ID.size());

    if (dvpp_in_->img_vec.size() < dvpp_in_->b_info.batch_size) {
        HIAI_ENGINE_LOG(
            HIAI_IDE_INFO, "[DvppEngine] Wait for other %d batch image info!",
            (dvpp_in_->b_info.batch_size - dvpp_in_->img_vec.size()));
        return HIAI_OK;
    }

    HIAI_ENGINE_LOG(HIAI_IDE_INFO, "[DvppEngine] dvpp_in addr: %lx, size: %d\n",
                    (unsigned long)dvpp_in_->img_vec[0].data.get(),
                    dvpp_in_->img_vec[0].size);

    dvpp_out_.reset();

    // process dvpp
    ret = HandleDvpp();
    if (HIAI_ERROR == ret) {
        HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "[DvppEngine]dvpp process error!");
        ClearData();
        return HIAI_ERROR;
    }

    // send to next engine after dvpp process
    if (!SendPreProcessData()) {
        HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "[DvppEngine]send dvpp result error!");
        ClearData();
        return HIAI_ERROR;
    }

    ClearData();
    HIAI_ENGINE_LOG(HIAI_IDE_INFO, "[DvppEngine] end process!");
    return HIAI_OK;
}

void DvppEngine::ClearData() {
    dvpp_in_.reset();
    dvpp_out_.reset();
}

bool DvppEngine::NeedCrop() {
    bool crop = true;
    if (-1 == (*dvpp_config_).point_x || -1 == (*dvpp_config_).point_y ||
        -1 == (*dvpp_config_).crop_width || -1 == (*dvpp_config_).crop_height) {
        crop = false;
    }
    return crop;
}

bool DvppEngine::ProcessCrop(VpcUserCropConfigure &area, const int &width,
                             const int &height, const int &real_width,
                             const int &real_height) {
    // default no crop
    int left_offset  = 0;               // the left side of the cropped image
    int right_offset = real_width - 1;  // the right side of the cropped image
    int up_offset    = 0;               // the top side of the cropped image
    uint32_t down_offset =
        real_height - 1;  // the bottom side of the cropped image
    int fixed_width  = dvpp_config_->crop_width;   // the actual crop width
    int fixed_height = dvpp_config_->crop_height;  // the actual crop height

    // user crop
    if (NeedCrop()) {
        HIAI_ENGINE_LOG(HIAI_IDE_INFO, "[DvppEngine] User crop, System crop");
        // amend input offset to avoid the processed range to exceed real range
        // of image
        if ((dvpp_config_->point_x + dvpp_config_->crop_width) > real_width) {
            fixed_width = real_width - dvpp_config_->point_x;
        }
        if ((dvpp_config_->point_y + dvpp_config_->crop_height) > real_height) {
            fixed_height = real_height - dvpp_config_->point_y;
        }

        left_offset  = dvpp_config_->point_x;
        right_offset = left_offset + fixed_width - 1;
        up_offset    = dvpp_config_->point_y;
        down_offset  = up_offset + fixed_height - 1;
        HIAI_ENGINE_LOG(HIAI_IDE_INFO,
                        "[DvppEngine] left_offset: %u, right_offset: %u, "
                        "up_offset: %u, down_offset: %u",
                        left_offset, right_offset, up_offset, down_offset);
    }

    // check param validity(range---max:4096*4096, min:16*16), left_offset and
    // up_offset cannot larger than real width and height -1 is the reason that
    // if point_x, point_y is (0, 0), it represent the 1*1 area.
    if (left_offset >= real_width || up_offset >= real_height ||
        right_offset - left_offset > DVPP_SUPPORT_MAX_WIDTH - 1 ||
        right_offset - left_offset < DVPP_SUPPORT_MIN_WIDTH - 1 ||
        down_offset - up_offset > DVPP_SUPPORT_MAX_HEIGHT - 1 ||
        down_offset - up_offset < DVPP_SUPPORT_MIN_HEIGHT - 1) {
        HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "crop range error");
        return false;
    }

    // restriction: left_offset and up_offset of inputputArea must be even,
    // right_offset and down_offset of inputputArea must be odd.
    area.leftOffset  = CHECK_EVEN(left_offset);
    area.rightOffset = CHECK_ODD(right_offset);
    area.upOffset    = CHECK_EVEN(up_offset);
    area.downOffset  = CHECK_ODD(down_offset);
    return true;
}

int DvppEngine::HandleDvpp() {
    if (nullptr == dvpp_in_ || nullptr == dvpp_in_.get()) {
        HIAI_ENGINE_LOG(HIAI_IDE_ERROR,
                        "[DvppEngine]__hanlde_dvpp, input data is null!");
        return HIAI_ERROR;
    }

    int i = 0;
    for (auto image_data : dvpp_in_->img_vec) {
        image_frame_id_ = dvpp_in_->b_info.frame_ID[i];

        if (IMAGE_TYPE_JPEG == (ImageTypeT)image_data.format) {
            if (HIAI_OK != HandleJpeg(image_data)) {
                HIAI_ENGINE_LOG(
                    HIAI_IDE_ERROR,
                    "[DvppEngine]Handle One Image Error, Continue Next");
            }
        } else if (IMAGE_TYPE_PNG == (ImageTypeT)image_data.format) {
            if (HIAI_OK != HandlePng(image_data)) {
                HIAI_ENGINE_LOG(
                    HIAI_IDE_ERROR,
                    "[DvppEngine]Handle One Image Error, Continue Next");
            }
        } else {  // default IMAGE_TYPE_NV12
            if (HIAI_OK != HandleVpc(image_data)) {
                HIAI_ENGINE_LOG(
                    HIAI_IDE_ERROR,
                    "[DvppEngine]Handle One Image Error, Continue Next");
            }
        }
        ++i;
    }

    return HIAI_OK;
}

// jpeg pic process flow:
//  1. DVPP_CTL_JPEGD_PROC
//  2. DVPP_CTL_TOOL_CASE_GET_RESIZE_PARAM
//  3. DVPP_CTL_VPC_PROC
int DvppEngine::HandleJpeg(const hiai::ImageData<uint8_t> &img) {
    if (nullptr == pidvppapi_) {
        HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "[DvppEngine] pidvppapi_ is null!\n");
        return HIAI_ERROR;
    }

    struct JpegdIn jpegd_in_data;  // input data
    // the pointer addr of jpeg pic data
    jpegd_in_data.jpegData = (unsigned char *)(img.data.get());
    // the size of jpeg pic data
    jpegd_in_data.jpegDataSize = img.size;
    //(*dvpp_config_).yuv420_need;true:output yuv420 data, otherwize:raw format.
    jpegd_in_data.isYUV420Need = false;
    // currently, only support V before U, reserved
    jpegd_in_data.isVBeforeU = true;

    struct JpegdOut jpegd_out_data;  // output data

    // create inputdata and outputdata for jpegd process
    dvppapi_ctl_msg dvppapi_ctlmsg;
    dvppapi_ctlmsg.in       = (void *)&jpegd_in_data;
    dvppapi_ctlmsg.in_size  = sizeof(struct JpegdIn);
    dvppapi_ctlmsg.out      = (void *)&jpegd_out_data;
    dvppapi_ctlmsg.out_size = sizeof(struct JpegdOut);

    if (0 != DvppCtl(pidvppapi_, DVPP_CTL_JPEGD_PROC, &dvppapi_ctlmsg)) {
        // if this single jpeg pic is processed with error, return directly, and
        // then process next pic if there any.
        HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "[DvppEngine] JPEGDERROR, FrameID:%u",
                        image_frame_id_);
        jpegd_out_data.cbFree();  // release memory from caller.
        return HIAI_ERROR;
    }

    int ret = HandleVpcWithParam(
        jpegd_out_data.yuvData, jpegd_out_data.imgWidthAligned,
        jpegd_out_data.imgHeightAligned, jpegd_out_data.yuvDataSize, img,
        FILE_TYPE_PIC_JPEG, jpegd_out_data.outFormat);
    jpegd_out_data.cbFree();  // release memory from caller.
    if (HIAI_OK != ret) {
        // if vpc process with error, return directly, and then process next pic
        // if there any.
        HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "[DvppEngine]VPCERROR, FrameID:%u",
                        image_frame_id_);
        return HIAI_ERROR;
    }
    return HIAI_OK;
}

// png pic process flow:
//  1. DVPP_CTL_PNGD_PROC
//  2. DVPP_CTL_TOOL_CASE_GET_RESIZE_PARAM
//  3. DVPP_CTL_VPC_PROC
int DvppEngine::HandlePng(const hiai::ImageData<uint8_t> &img) {
    if (nullptr == pidvppapi_) {
        HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "[DvppEngine] pidvppapi is null");
        return HIAI_ERROR;
    }

    struct PngInputInfoAPI input_pngdata;  // input data
                                           // input data pointer addr
    input_pngdata.inputData = (void *)(img.data.get());
    // input data length
    input_pngdata.inputSize = img.size;
    // whether to transform
    input_pngdata.transformFlag = (*dvpp_config_).transform_flag;

    struct PngOutputInfoAPI output_pngdata;  // output data

    dvppapi_ctl_msg dvppapi_ctlmsg;
    dvppapi_ctlmsg.in       = (void *)(&input_pngdata);
    dvppapi_ctlmsg.in_size  = sizeof(struct PngInputInfoAPI);
    dvppapi_ctlmsg.out      = (void *)(&output_pngdata);
    dvppapi_ctlmsg.out_size = sizeof(struct PngOutputInfoAPI);

    if (0 != DvppCtl(pidvppapi_, DVPP_CTL_PNGD_PROC, &dvppapi_ctlmsg)) {
        // if this single jpeg pic is processed with error, return directly, and
        // process next pic
        HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "[DvppEngine]PNGDERROR, FrameID:%u",
                        image_frame_id_);
        if (output_pngdata.address != nullptr) {
            // release memory from caller.
            munmap(
                output_pngdata.address,
                ALIGN_UP(output_pngdata.size + VPC_OUT_WIDTH_STRIDE, MAP_2M));
            output_pngdata.address = nullptr;
        }
        return HIAI_ERROR;
    }

    int ret = HandleVpcWithParam(
        (unsigned char *)output_pngdata.outputData, output_pngdata.widthAlign,
        output_pngdata.highAlign, output_pngdata.outputSize, img,
        FILE_TYPE_PIC_PNG, output_pngdata.format);

    if (output_pngdata.address != nullptr) {
        // release memory from caller.
        munmap(output_pngdata.address,
               ALIGN_UP(output_pngdata.size + VPC_OUT_WIDTH_STRIDE, MAP_2M));
        output_pngdata.address = nullptr;
    }
    if (HIAI_OK != ret) {
        // if vpc process with error, return directly, and then process next.
        HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "[DvppEngine]VPCERROR, FrameID:%u",
                        image_frame_id_);
        return HIAI_ERROR;
    }
    return HIAI_OK;
}

int DvppEngine::HandleVpc(const hiai::ImageData<uint8_t> &img) {
    if (nullptr == img.data || nullptr == img.data.get()) {
        HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "[DvppEngine]img.data is null, ERROR");
        return HIAI_ERROR;
    }

    if (IMAGE_TYPE_NV12 != (ImageTypeT)img.format) {
        HIAI_ENGINE_LOG(HIAI_IDE_ERROR,
                        "[DvppEngine]the format is not yuv, ERROR");
        return HIAI_ERROR;
    }

    unsigned char *align_buffer = img.data.get();
    int align_width             = img.width;
    int align_height            = img.height;
    // the size of yuv data is 1.5 times of width*height
    int align_image_len  = align_width * align_height * 3 / 2;
    bool align_mmap_flag = false;

    // if width or height is not align, copy memory
    if (0 != (img.width % VPC_OUT_WIDTH_STRIDE) ||
        0 != (img.height % VPC_OUT_HIGH_STRIDE)) {
        align_mmap_flag = true;
        align_width     = ALIGN_UP(img.width, VPC_OUT_WIDTH_STRIDE);
        align_height    = ALIGN_UP(img.height, VPC_OUT_HIGH_STRIDE);
        align_image_len = align_width * align_height * 3 / 2;
        align_buffer    = (unsigned char *)HIAI_DVPP_DMalloc(align_image_len);
        if (nullptr == align_buffer) {
            HIAI_ENGINE_LOG(
                HIAI_IDE_ERROR,
                "[DvppEngine]HIAI_DVPP_DMalloc align_buffer is null, ERROR");
            return HIAI_ERROR;
        }
        for (unsigned int i = 0; i < img.height; i++) {
            int ret = memcpy_s(align_buffer + i * align_width, align_width,
                               img.data.get() + i * img.width, img.width);
            if (0 != ret) {
                HIAI_ENGINE_LOG(HIAI_IDE_ERROR,
                                "[DvppEngine]memcpy_s error in copy Y");
                HIAI_DVPP_DFree(align_buffer);
                align_buffer = nullptr;
                return HIAI_ERROR;
            }
        }
        for (unsigned int i = 0; i < img.height / 2; i++) {
            int ret = memcpy_s(
                align_buffer + i * align_width + align_width * align_height,
                align_width,
                img.data.get() + i * img.width + img.width * img.height,
                img.width);
            if (0 != ret) {
                HIAI_ENGINE_LOG(HIAI_IDE_ERROR,
                                "[DvppEngine]memcpy_s error in copy UV");
                HIAI_DVPP_DFree(align_buffer);
                align_buffer = nullptr;
                return HIAI_ERROR;
            }
        }
    }

    int ret = HandleVpcWithParam(align_buffer, align_width, align_height,
                                 align_image_len, img, FILE_TYPE_YUV, 3);
    if (HIAI_OK != ret) {
        // if vpc process with error, return directly, and then process next.
        HIAI_ENGINE_LOG(HIAI_IDE_ERROR,
                        "[DvppEngine] call DVPP_CTL_VPC_PROC process faild");
        if (align_mmap_flag) {  // release memory
            HIAI_DVPP_DFree(align_buffer);
            align_buffer = nullptr;
        }
        return HIAI_ERROR;
    }
    if (align_mmap_flag) {  // release memory
        HIAI_DVPP_DFree(align_buffer);
        align_buffer = nullptr;
    }
    return HIAI_OK;
}

int DvppEngine::HandleVpcWithParam(const unsigned char *buffer,
                                   const int &width, const int &height,
                                   const long &bufferSize,
                                   const hiai::ImageData<uint8_t> &img,
                                   const FILE_TYPE &type, const int &format) {
    int real_width  = img.width;
    int real_height = img.height;

    HIAI_ENGINE_LOG(HIAI_IDE_INFO,
                    "[DvppEngine] real_width: %d, real_height: %d, width: %d, "
                    "height: %d, format: %d",
                    real_width, real_height, width, height, format);

    if (nullptr == pidvppapi_ || nullptr == buffer) {
        HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "[DvppEngine]pidvppapi_ is null");
        return HIAI_ERROR;
    }

    VpcUserImageConfigure user_image;
    string para_set_path[1];
    if (type == FILE_TYPE_PIC_JPEG) {
        // format is responding to color sapce after decoder, which is needed to
        // match the input color space of vpc
        switch (format) {
            case DVPP_JPEG_DECODE_OUT_YUV444:
                user_image.inputFormat = INPUT_YUV444_SEMI_PLANNER_VU;
                break;
            case DVPP_JPEG_DECODE_OUT_YUV422_H2V1:
                user_image.inputFormat = INPUT_YUV422_SEMI_PLANNER_VU;
                break;
            case DVPP_JPEG_DECODE_OUT_YUV420:
                user_image.inputFormat = INPUT_YUV420_SEMI_PLANNER_VU;
                break;
            case DVPP_JPEG_DECODE_OUT_YUV400:
                user_image.inputFormat = INPUT_YUV400;
                break;
            default:
                HIAI_ENGINE_LOG(HIAI_IDE_ERROR,
                                "[DvppEngine]Jpegd out format[%d] is error",
                                format);
                break;
        }
    } else if (type == FILE_TYPE_PIC_PNG) {
        switch (format) {
            case DVPP_PNG_DECODE_OUT_RGB:
                user_image.inputFormat = INPUT_RGB;
                break;
            case DVPP_PNG_DECODE_OUT_RGBA:
                user_image.inputFormat = INPUT_RGBA;
                break;
            default:
                HIAI_ENGINE_LOG(HIAI_IDE_ERROR,
                                "[DvppEngine]Pngd out format[%d] is error",
                                format);
                break;
        }
    } else {
        user_image.inputFormat = INPUT_YUV420_SEMI_PLANNER_UV;
    }
    user_image.widthStride        = width;
    user_image.heightStride       = height;
    user_image.outputFormat       = OUTPUT_YUV420SP_UV;
    user_image.bareDataAddr       = (uint8_t *)buffer;
    user_image.bareDataBufferSize = bufferSize;

    VpcUserRoiConfigure roi_configure;
    roi_configure.next                        = nullptr;
    VpcUserRoiInputConfigure *input_configure = &roi_configure.inputConfigure;
    input_configure->cropArea.leftOffset      = 0;
    input_configure->cropArea.rightOffset     = real_width - 1;
    input_configure->cropArea.upOffset        = 0;
    input_configure->cropArea.downOffset      = real_height - 1;

    uint32_t resize_width  = (uint32_t)dvpp_config_->resize_width;
    uint32_t resize_height = (uint32_t)dvpp_config_->resize_height;
    if (0 == resize_width || 0 == resize_height) {
        HIAI_ENGINE_LOG(HIAI_IDE_INFO,
                        "[DvppEngine] user donnot need resize, resize "
                        "width/height use real size of pic");
        resize_width  = real_width;
        resize_height = real_height;
    }

    if (resize_width > DVPP_SUPPORT_MAX_WIDTH ||
        resize_width < DVPP_SUPPORT_MIN_WIDTH ||
        resize_height > DVPP_SUPPORT_MAX_HEIGHT ||
        resize_height < DVPP_SUPPORT_MIN_HEIGHT) {
        HIAI_ENGINE_LOG(
            HIAI_IDE_ERROR,
            "[DvppEngine]resize range error, resize_width:%u, resize_height:%u",
            resize_width, resize_height);
        return HIAI_ERROR;
    }

    VpcUserRoiOutputConfigure *output_configure =
        &roi_configure.outputConfigure;
    if (ProcessCrop(input_configure->cropArea, width, height, real_width,
                    real_height)) {
        // restriction: leftOffset and upOffset of outputArea must be even,
        // rightOffset and downOffset of outputArea must be odd.
        output_configure->outputArea.leftOffset  = 0;
        output_configure->outputArea.rightOffset = CHECK_ODD(resize_width - 1);
        output_configure->outputArea.upOffset    = 0;
        output_configure->outputArea.downOffset  = CHECK_ODD(resize_height - 1);
        output_configure->widthStride =
            ALIGN_UP(resize_width, VPC_OUT_WIDTH_STRIDE);
        output_configure->heightStride =
            ALIGN_UP(resize_height, VPC_OUT_HIGH_STRIDE);
        output_configure->bufferSize = output_configure->widthStride *
                                       output_configure->heightStride * 3 / 2;
        output_configure->addr = static_cast<uint8_t *>(
            HIAI_DVPP_DMalloc(output_configure->bufferSize));
        memset(output_configure->addr, 0, output_configure->bufferSize);
        if (nullptr == output_configure->addr) {
            HIAI_ENGINE_LOG(HIAI_IDE_ERROR,
                            "[DvppEngine]output_configure->addr is null");
            return HIAI_ERROR;
        }
        HIAI_ENGINE_LOG(
            HIAI_IDE_INFO,
            "[DvppEngine]input_configure cropArea:%u, %u, %u, %u, "
            "output_configure outputArea:%u, %u, %u, %u, stride:%u, %u, %u",
            input_configure->cropArea.leftOffset,
            input_configure->cropArea.rightOffset,
            input_configure->cropArea.upOffset,
            input_configure->cropArea.downOffset,
            output_configure->outputArea.leftOffset,
            output_configure->outputArea.rightOffset,
            output_configure->outputArea.upOffset,
            output_configure->outputArea.downOffset,
            output_configure->widthStride, output_configure->heightStride,
            output_configure->bufferSize);
    } else {
        HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "[DvppEngine]ProcessCrop error");
        return HIAI_ERROR;
    }
    user_image.roiConfigure = &roi_configure;

    if (!dvpp_config_->dvpp_para.empty()) {
        para_set_path[0] += dvpp_config_->dvpp_para.c_str();
        user_image.yuvScalerParaSetAddr =
            reinterpret_cast<uint64_t>(para_set_path);
        user_image.yuvScalerParaSetSize  = 1;
        user_image.yuvScalerParaSetIndex = 0;
        HIAI_ENGINE_LOG(HIAI_IDE_INFO, "[DvppEngine]dvpp_para:%s",
                        dvpp_config_->dvpp_para.c_str());
    }

    dvppapi_ctl_msg dvppapi_ctlmsg;
    dvppapi_ctlmsg.in      = (void *)&user_image;
    dvppapi_ctlmsg.in_size = sizeof(VpcUserImageConfigure);
    if (0 != DvppCtl(pidvppapi_, DVPP_CTL_VPC_PROC, &dvppapi_ctlmsg)) {
        HIAI_ENGINE_LOG(HIAI_IDE_ERROR,
                        "[DvppEngine] call dvppctl process:VPC faild");
        HIAI_DVPP_DFree(output_configure->addr);
        output_configure->addr = nullptr;
        return HIAI_ERROR;
    }

    if (nullptr == dvpp_out_) {
        dvpp_out_ = std::make_shared<DvppInputDataType>();
        if (nullptr == dvpp_out_ || nullptr == dvpp_out_.get()) {
            HIAI_ENGINE_LOG(HIAI_IDE_ERROR,
                            "[DvppEngine] call dvpp_out_ make_shared failed");
            HIAI_DVPP_DFree(output_configure->addr);
            output_configure->addr = nullptr;
            return HIAI_ERROR;
        }
        dvpp_out_->output_info = dvpp_in_->output_info;
        dvpp_out_->b_info      = dvpp_in_->b_info;
        dvpp_out_->b_info.frame_ID.clear();
    }

    hiai::ImageData<uint8_t> image_data;
    image_data.width = output_configure->outputArea.rightOffset -
                       output_configure->outputArea.leftOffset + 1;
    image_data.height = output_configure->outputArea.downOffset -
                        output_configure->outputArea.upOffset + 1;
    image_data.size    = output_configure->bufferSize;
    image_data.channel = img.channel;
    image_data.format  = img.format;

    uint8_t *tmp = nullptr;
    try {
        tmp = new uint8_t[image_data.size];
    } catch (const std::bad_alloc &e) {
        HIAI_ENGINE_LOG(
            HIAI_IDE_ERROR,
            "[DvppEngine] failed to allocate buffer for out_buffer");
        return HIAI_ERROR;
    }
    std::shared_ptr<uint8_t> out_buffer(tmp, [](uint8_t *p) { delete[] p; });
    if (nullptr == out_buffer || nullptr == out_buffer.get()) {
        HIAI_ENGINE_LOG(HIAI_IDE_ERROR,
                        "[DvppEngine] call out_buffer shared_ptr failed");
        HIAI_DVPP_DFree(output_configure->addr);
        output_configure->addr = nullptr;
        return HIAI_ERROR;
    }
    int ret = memcpy_s(out_buffer.get(), image_data.size,
                       output_configure->addr, output_configure->bufferSize);

    HIAI_DVPP_DFree(output_configure->addr);
    output_configure->addr = nullptr;
    if (ret != 0) {
        HIAI_ENGINE_LOG(HIAI_IDE_ERROR,
                        "[DvppEngine] memcpy_s out buffer data error");
        out_buffer = nullptr;
        return HIAI_ERROR;
    }
    image_data.data = out_buffer;
    dvpp_out_->img_vec.push_back(image_data);
    dvpp_out_->b_info.frame_ID.push_back(image_frame_id_);

    return HIAI_OK;
}

bool DvppEngine::SendPreProcessData() {
    if (nullptr == dvpp_out_) {
        HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "[DvppEngine]Nothing to send!");
        return false;
    }

    if (dvpp_out_->img_vec.empty() ||
        dvpp_out_->img_vec.size() != dvpp_out_->b_info.batch_size) {
        HIAI_ENGINE_LOG(HIAI_IDE_ERROR,
                        "[DvppEngine]input size (%u) not match with batch (%u)",
                        dvpp_out_->img_vec.size(),
                        dvpp_out_->b_info.batch_size);
        return false;
    }

    HIAI_StatusT ret = HIAI_OK;
    uint32_t size_in_bytes =
        dvpp_out_->img_vec[0].size * dvpp_out_->b_info.batch_size;
    uint8_t *buffer = nullptr;
    ret = hiai::HIAIMemory::HIAI_DMalloc(size_in_bytes, (void *&)buffer);
    if (HIAI_OK != ret || nullptr == buffer) {
        HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "[DvppEngine] DMalloc buffer error!\n");
        return false;
    }

    int offset = 0;
    for (auto image_data : dvpp_out_->img_vec) {
        memcpy(buffer + offset, image_data.data.get(), image_data.size);
        offset += image_data.size;
    }

    std::shared_ptr<TransferDataType> trans_data =
        std::make_shared<TransferDataType>();
    trans_data->info.cmd_type      = CT_DataTransfer;
    trans_data->info.size_in_bytes = size_in_bytes;
    trans_data->output_info        = dvpp_out_->output_info;
    trans_data->data_len           = size_in_bytes;
    trans_data->data.reset(buffer, [](uint8_t *) {});

    ret = SendData(0, "TransferDataType",
                   std::static_pointer_cast<void>(trans_data));
    if (HIAI_OK != ret) {
        HIAI_ENGINE_LOG(HIAI_IDE_ERROR, "DvppEngine send data failed\n");
        return false;
    }

    return true;
}

}  // namespace TNN_NS
