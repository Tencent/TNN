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

#include "tnn/device/atlas/atlas_mat_converter.h"
#include "tnn/core/macro.h"
#include "tnn/device/atlas/atlas_runtime.h"
#include "tnn/device/atlas/atlas_utils.h"
#include "tnn/utils/data_format_converter.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

// default contructor will create convert buffer
AtlasMatConverterAcc::AtlasMatConverterAcc() {
    input_desc_ = acldvppCreatePicDesc();
    if (nullptr == input_desc_) {
        LOGE("create input desc for mat converter failed\n");
        return;
    }

    output_desc_ = acldvppCreatePicDesc();
    if (nullptr == input_desc_) {
        LOGE("create input desc for mat converter failed\n");
        return;
    }

    dvpp_channel_desc_ = acldvppCreateChannelDesc();
    if (nullptr == dvpp_channel_desc_) {
        LOGE("create channel desc for mat converter failed\n");
        return;
    }

    aclError ret = acldvppCreateChannel(dvpp_channel_desc_);
    if (ACL_ERROR_NONE != ret) {
        LOGE("create channel for mat converter failed\n");
        return;
    }

    init_success_ = true;
}

AtlasMatConverterAcc::~AtlasMatConverterAcc() {
    aclError ret;
    if (nullptr != dvpp_input_buffer_) {
        ret = acldvppFree(dvpp_input_buffer_);
        if (ACL_ERROR_NONE != ret) {
            LOGE("acldvppFree failed, ret = %d\n", ret);
        }
        dvpp_input_buffer_ = nullptr;
        input_buffer_size_ = 0;
    }

    if (nullptr != dvpp_output_buffer_) {
        ret = acldvppFree(dvpp_output_buffer_);
        if (ACL_ERROR_NONE != ret) {
            LOGE("acldvppFree failed, ret = %d\n", ret);
        }
        dvpp_output_buffer_ = nullptr;
        output_buffer_size_ = 0;
    }

    if (nullptr != input_desc_) {
        ret = acldvppDestroyPicDesc(input_desc_);
        if (ACL_ERROR_NONE != ret) {
            LOGE("acldvppDestroyPicDesc failed, ret = %d\n", ret);
        }
        input_desc_ = nullptr;
    }

    if (nullptr != output_desc_) {
        ret = acldvppDestroyPicDesc(output_desc_);
        if (ACL_ERROR_NONE != ret) {
            LOGE("acldvppDestroyPicDesc failed, ret = %d\n", ret);
        }
        output_desc_ = nullptr;
    }

    if (nullptr != dvpp_channel_desc_) {
        ret = acldvppDestroyChannel(dvpp_channel_desc_);
        if (ACL_ERROR_NONE != ret) {
            LOGE("acldvppDestroyChannel failed, ret = %d\n", ret);
        }

        ret = acldvppDestroyChannelDesc(dvpp_channel_desc_);
        if (ACL_ERROR_NONE != ret) {
            LOGE("acldvppDestroyChannelDesc failed, ret = %d\n", ret);
        }
        dvpp_channel_desc_ = nullptr;
    }
}

Status AtlasMatConverterAcc::Copy(Mat& src, Mat& dst, void* command_queue) {
    if (!init_success_) {
        LOGE("init mat converter failed!\n");
        return Status(TNNERR_NULL_PARAM, "init mat converter failed!");
    }

    auto atlas_cmd_queue = static_cast<AtlasCommandQueue*>(command_queue);
    if (atlas_cmd_queue == nullptr) {
        LOGE("get atlas command queue failed!\n");
        return Status(TNNERR_NULL_PARAM, "get atlas command queue failed!");
    }

    aclrtMemcpyKind memcpy_type;
    if (DEVICE_ATLAS == src.GetDeviceType() && DEVICE_ATLAS == dst.GetDeviceType()) {
        memcpy_type = ACL_MEMCPY_DEVICE_TO_DEVICE;
    } else if (DEVICE_ATLAS == src.GetDeviceType() &&
               (DEVICE_NAIVE == dst.GetDeviceType() || DEVICE_ARM == dst.GetDeviceType())) {
        memcpy_type = ACL_MEMCPY_DEVICE_TO_HOST;
    } else if ((DEVICE_NAIVE == src.GetDeviceType() || DEVICE_ARM == src.GetDeviceType()) &&
               DEVICE_ATLAS == dst.GetDeviceType()) {
        memcpy_type = ACL_MEMCPY_HOST_TO_DEVICE;
    } else {
        return Status(TNNERR_ATLAS_DVPP_NOT_SUPPORT, "invalid mat device type for atlas Copy()");
    }

    Status tnn_ret = TNN_OK;
    int src_size   = 0;
    int dst_size   = 0;

    tnn_ret = MatUtils::GetMatByteSize(src, src_size);
    if (TNN_OK != tnn_ret) {
        return tnn_ret;
    }
    tnn_ret = MatUtils::GetMatByteSize(dst, dst_size);
    if (TNN_OK != tnn_ret) {
        return tnn_ret;
    }

    if (dst_size != src_size) {
        LOGE("invalid size for MatCopy\n");
        return Status(TNNERR_PARAM_ERR, "invalid size for MatCopy");
    }

    aclError acl_ret = aclrtMemcpy(dst.GetData(), src_size, src.GetData(), src_size, memcpy_type);
    if (ACL_ERROR_NONE != acl_ret) {
        return Status(TNNERR_ATLAS_RUNTIME_ERROR, "acl memory copy failed");
    }

    return TNN_OK;
}

Status AtlasMatConverterAcc::Resize(Mat& src, Mat& dst, ResizeParam param, void* command_queue) {
    if (!init_success_) {
        LOGE("init mat converter failed!\n");
        return Status(TNNERR_NULL_PARAM, "init mat converter failed!");
    }

    auto atlas_cmd_queue = static_cast<AtlasCommandQueue*>(command_queue);
    if (atlas_cmd_queue == nullptr) {
        LOGE("get atlas command queue failed!\n");
        return Status(TNNERR_NULL_PARAM, "get atlas command queue failed!");
    }

    if (0 != param.scale_w && 0 != param.scale_h) {
        int dst_width  = src.GetWidth() * param.scale_w;
        int dst_height = src.GetHeight() * param.scale_h;

        if (dst_width != dst.GetWidth() || dst_height != dst.GetHeight()) {
            dst = Mat(dst.GetDeviceType(), dst.GetMatType(), {dst.GetBatch(), dst.GetChannel(), dst_height, dst_width},
                      nullptr);
        }
    } else if (0 == dst.GetWidth() || 0 == dst.GetHeight()) {
        LOGE("dst mat size is invailed! (%dx%d)\n", dst.GetWidth(), dst.GetHeight());
        return Status(TNNERR_NULL_PARAM, "resize param is invalid!");
    }

    Status ret       = TNN_OK;
    aclError acl_ret = ACL_ERROR_NONE;

    ret = PrepareInput(src);
    if (TNN_OK != ret) {
        return ret;
    }

    ret = PrepareOutput(dst);
    if (TNN_OK != ret) {
        return ret;
    }

    acldvppResizeConfig* resize_config = acldvppCreateResizeConfig();
    if (nullptr == resize_config) {
        LOGE("create resize config for mat converter failed\n");
        return Status(TNNERR_ATLAS_RUNTIME_ERROR, "acldvppCreateResizeConfig failed");
    }

    acl_ret =
        acldvppVpcResizeAsync(dvpp_channel_desc_, input_desc_, output_desc_, resize_config, atlas_cmd_queue->stream);
    if (ACL_ERROR_NONE != acl_ret) {
        LOGE("acldvppVpcResizeAsync failed, ret = %d\n", acl_ret);
        return Status(TNNERR_ATLAS_RUNTIME_ERROR, "acldvppVpcResizeAsync failed");
    }

    if (nullptr != resize_config) {
        acl_ret = acldvppDestroyResizeConfig(resize_config);
        if (ACL_ERROR_NONE != acl_ret) {
            LOGE("acldvppDestroyResizeConfig failed, ret = %d\n", acl_ret);
        }
        resize_config = nullptr;
    }

    LOGD("Stream ID: 0x%lx\n", atlas_cmd_queue->stream);
    acl_ret = aclrtSynchronizeStream(atlas_cmd_queue->stream);
    if (ACL_ERROR_NONE != acl_ret) {
        LOGE("aclrtSynchronizeStream failed, ret = %d\n", acl_ret);
        return Status(TNNERR_ATLAS_RUNTIME_ERROR, "aclrtSynchronizeStream failed");
    }

    ret = ProcessOutput(dst);
    if (TNN_OK != ret) {
        return ret;
    }

    return TNN_OK;
}

Status AtlasMatConverterAcc::Crop(Mat& src, Mat& dst, CropParam param, void* command_queue) {
    if (!init_success_) {
        LOGE("init mat converter failed!\n");
        return Status(TNNERR_NULL_PARAM, "init mat converter failed!");
    }

    auto atlas_cmd_queue = static_cast<AtlasCommandQueue*>(command_queue);
    if (atlas_cmd_queue == nullptr) {
        LOGE("get atlas command queue failed!\n");
        return Status(TNNERR_NULL_PARAM, "get atlas command queue failed!");
    }

    Status ret = TNN_OK;
    ret        = PrepareInput(src);
    if (TNN_OK != ret) {
        return ret;
    }

    CropParam param_real = ProcessCropParam(param);
    dst                  = Mat(dst.GetDeviceType(), dst.GetMatType(),
                               {dst.GetBatch(), dst.GetChannel(), 
                                param_real.height, param_real.width}, nullptr);
    ret                  = PrepareOutput(dst);
    if (TNN_OK != ret) {
        return ret;
    }

    acldvppRoiConfig* crop_roi_config =
        acldvppCreateRoiConfig(param_real.top_left_x, param_real.top_left_x + param_real.width - 1,
                               param_real.top_left_y, param_real.top_left_y + param_real.height - 1);
    if (nullptr == crop_roi_config) {
        LOGE("create crop roi config in crop failed\n");
        return Status(TNNERR_ATLAS_RUNTIME_ERROR, "acldvppCreateRoiConfig failed");
    }

    aclError acl_ret = ACL_ERROR_NONE;
    acl_ret =
        acldvppVpcCropAsync(dvpp_channel_desc_, input_desc_, output_desc_, crop_roi_config, atlas_cmd_queue->stream);
    if (ACL_ERROR_NONE != acl_ret) {
        LOGE("acldvppVpcResizeAsync failed, ret = %d\n", acl_ret);
        return Status(TNNERR_ATLAS_RUNTIME_ERROR, "acldvppVpcResizeAsync failed");
    }

    LOGD("Stream ID: 0x%lx\n", atlas_cmd_queue->stream);
    acl_ret = aclrtSynchronizeStream(atlas_cmd_queue->stream);
    if (ACL_ERROR_NONE != acl_ret) {
        LOGE("aclrtSynchronizeStream failed, ret = %d\n", acl_ret);
        return Status(TNNERR_ATLAS_RUNTIME_ERROR, "aclrtSynchronizeStream failed");
    }

    ret = ProcessOutput(dst);
    if (TNN_OK != ret) {
        return ret;
    }

    if (nullptr != crop_roi_config) {
        acl_ret = acldvppDestroyRoiConfig(crop_roi_config);
        if (ACL_ERROR_NONE != acl_ret) {
            LOGE("acldvppDestroyRoiConfig failed, ret = %d\n", acl_ret);
            return Status(TNNERR_ATLAS_RUNTIME_ERROR, "acldvppDestroyRoiConfig failed");
        }
    }

    return TNN_OK;
}

Status AtlasMatConverterAcc::WarpAffine(Mat& src, Mat& dst, WarpAffineParam param, void* command_queue) {
    if (!init_success_) {
        LOGE("init mat converter failed!\n");
        return Status(TNNERR_NULL_PARAM, "init mat converter failed!");
    }

    auto atlas_cmd_queue = static_cast<AtlasCommandQueue*>(command_queue);
    if (atlas_cmd_queue == nullptr) {
        LOGE("get atlas command queue failed!\n");
        return Status(TNNERR_NULL_PARAM, "get atlas command queue failed!");
    }

    return Status(TNNERR_ATLAS_DVPP_NOT_SUPPORT, "atlas mat not support WarpAffine");
}

Status AtlasMatConverterAcc::CvtColor(Mat& src, Mat& dst, ColorConversionType type, void* command_queue) {
    if (!init_success_) {
        LOGE("init mat converter failed!\n");
        return Status(TNNERR_NULL_PARAM, "init mat converter failed!");
    }

    auto atlas_cmd_queue = static_cast<AtlasCommandQueue*>(command_queue);
    if (atlas_cmd_queue == nullptr) {
        LOGE("get atlas command queue failed!\n");
        return Status(TNNERR_NULL_PARAM, "get atlas command queue failed!");
    }

    return Status(TNNERR_ATLAS_DVPP_NOT_SUPPORT, "atlas mat not support CvtColor");
}

Status AtlasMatConverterAcc::CopyMakeBorder(Mat& src, Mat& dst, CopyMakeBorderParam param, void* command_queue) {
    if (!init_success_) {
        LOGE("init mat converter failed!\n");
        return Status(TNNERR_NULL_PARAM, "init mat converter failed!");
    }

    auto atlas_cmd_queue = static_cast<AtlasCommandQueue*>(command_queue);
    if (atlas_cmd_queue == nullptr) {
        LOGE("get atlas command queue failed!\n");
        return Status(TNNERR_NULL_PARAM, "get atlas command queue failed!");
    }

    return Status(TNNERR_ATLAS_DVPP_NOT_SUPPORT, "atlas mat not support CopyMakeBorder");
}

Status AtlasMatConverterAcc::PrepareInput(Mat& mat) {
    int batch = mat.GetBatch();
    if (1 != batch) {
        LOGE("mat resize not support multi batch (batch is %d)!\n", batch);
        return Status(TNNERR_ATLAS_DVPP_NOT_SUPPORT, "atlas mat resize not support multi batch");
    }

    aclError acl_ret;
    Status tnn_ret;

    int width_aligned  = 0;
    int height_aligned = 0;
    int buffer_size    = 0;
    tnn_ret            = GetAlignedBufferSize(mat, 16, 2, buffer_size, width_aligned, height_aligned);
    if (TNN_OK != tnn_ret) {
        return tnn_ret;
    }

    LOGD("input width: %d  height: %d   width_aligned: %d  height_aligned: %d  buffer_size: %d\n", mat.GetWidth(),
         mat.GetHeight(), width_aligned, height_aligned, buffer_size);

    DeviceType device_type = mat.GetDeviceType();
    if (DEVICE_ATLAS == device_type) {
        LOGD("input is on device\n");
        // input device memory must by aligned with 16x2
        dvpp_input_buffer_ptr_ = mat.GetData();
    } else if (DEVICE_NAIVE == device_type || DEVICE_ARM == device_type) {
        LOGD("input is on host\n");
        // malloc device memory
        tnn_ret = MallocDeviceMemory(&dvpp_input_buffer_, input_buffer_size_, buffer_size);
        if (TNN_OK != tnn_ret) {
            return tnn_ret;
        }

        // copy from host to device
        tnn_ret = CopyFromHostToDeviceAligned(mat, dvpp_input_buffer_, 16, 2);
        if (TNN_OK != tnn_ret) {
            return tnn_ret;
        }

        dvpp_input_buffer_ptr_ = dvpp_input_buffer_;
    } else {
        LOGE("mat resize not support this input device type (device type is %d)!\n", device_type);
        return Status(TNNERR_ATLAS_DVPP_NOT_SUPPORT, "atlas mat resize not support this input device type");
    }

    acldvppPixelFormat dvpp_pixel_format;
    tnn_ret = ConvertFromMatTypeToDvppPixelFormat(mat.GetMatType(), dvpp_pixel_format);
    if (TNN_OK != tnn_ret) {
        return tnn_ret;
    }

    int width_stride = GetWidthStride(mat.GetMatType(), width_aligned);
    LOGD(
        "input set:   buffer addr: 0x%lx    dvpp format: %d   width: %d  height: %d  wight_stride: %d  height_stride: "
        "%d  buffer size: %d\n",
        dvpp_input_buffer_ptr_, dvpp_pixel_format, mat.GetWidth(), mat.GetHeight(), width_stride, height_aligned,
        buffer_size);
    acldvppSetPicDescData(input_desc_, dvpp_input_buffer_ptr_);
    acldvppSetPicDescFormat(input_desc_, dvpp_pixel_format);
    acldvppSetPicDescWidth(input_desc_, mat.GetWidth());
    acldvppSetPicDescHeight(input_desc_, mat.GetHeight());
    acldvppSetPicDescWidthStride(input_desc_, width_stride);
    acldvppSetPicDescHeightStride(input_desc_, height_aligned);
    acldvppSetPicDescSize(input_desc_, buffer_size);

    return TNN_OK;
}

Status AtlasMatConverterAcc::PrepareOutput(Mat& mat, int pad_value) {
    int batch = mat.GetBatch();
    if (1 != batch) {
        LOGE("atlas mat convert not support multi batch (batch is %d)!\n", batch);
        return Status(TNNERR_ATLAS_DVPP_NOT_SUPPORT, "atlas mat resize not support multi batch");
    }

    MatType mat_type = mat.GetMatType();
    if (NNV12 != mat_type && NNV21 != mat_type) {
        LOGE("atlas mat convert output only support NV12 or NV21!\n");
        return Status(TNNERR_ATLAS_DVPP_NOT_SUPPORT, "atlas mat convert not support this mat type");
    }

    aclError acl_ret;
    Status tnn_ret;

    int width_origin  = mat.GetWidth();
    int height_origin = mat.GetHeight();

    int width_aligned  = 0;
    int height_aligned = 0;
    int buffer_size    = 0;
    tnn_ret            = GetAlignedBufferSize(mat, 16, 2, buffer_size, width_aligned, height_aligned);
    if (TNN_OK != tnn_ret) {
        return tnn_ret;
    }
    LOGD("output width: %d  height: %d   width_aligned: %d  height_aligned: %d  buffer_size: %d\n", mat.GetWidth(),
         mat.GetHeight(), width_aligned, height_aligned, buffer_size);

    DeviceType device_type = mat.GetDeviceType();
    if (nullptr == mat.GetData()) {
        mat = Mat(device_type, mat.GetMatType(), {mat.GetBatch(), mat.GetChannel(), height_aligned, width_aligned});
    }

    // get dvpp_output_buffer
    if (DEVICE_ATLAS == device_type) {
        LOGD("output is on device\n");
        // output device memory must by aligned with 16x2
        dvpp_output_buffer_ptr_ = mat.GetData();
    } else if (DEVICE_NAIVE == device_type || DEVICE_ARM == device_type) {
        LOGD("output is on cpu\n");
        // malloc device memory
        tnn_ret = MallocDeviceMemory(&dvpp_output_buffer_, output_buffer_size_, buffer_size);
        if (TNN_OK != tnn_ret) {
            return tnn_ret;
        }

        dvpp_output_buffer_ptr_ = dvpp_output_buffer_;
    } else {
        LOGE("mat resize not support this input device type (device type is %d)!\n", device_type);
        return Status(TNNERR_ATLAS_DVPP_NOT_SUPPORT, "atlas mat resize not support this input device type");
    }
    acl_ret = aclrtMemset(dvpp_output_buffer_ptr_, buffer_size, pad_value, buffer_size);
    if (ACL_ERROR_NONE != acl_ret) {
        LOGE("aclrtMemset failed, ret = %d\n", acl_ret);
        return Status(TNNERR_ATLAS_RUNTIME_ERROR, "aclrtMemset failed");
    }

    acldvppPixelFormat dvpp_pixel_format;
    tnn_ret = ConvertFromMatTypeToDvppPixelFormat(mat.GetMatType(), dvpp_pixel_format);
    if (TNN_OK != tnn_ret) {
        return tnn_ret;
    }

    int width_stride = GetWidthStride(mat.GetMatType(), width_aligned);
    LOGD(
        "output set:   buffer addr: 0x%lx    dvpp format: %d   width: %d  height: %d  wight_stride: %d  height_stride: "
        "%d  buffer size: %d\n",
        dvpp_output_buffer_ptr_, dvpp_pixel_format, mat.GetWidth(), mat.GetHeight(), width_stride, height_aligned,
        buffer_size);
    acldvppSetPicDescData(output_desc_, dvpp_output_buffer_ptr_);
    acldvppSetPicDescFormat(output_desc_, dvpp_pixel_format);
    acldvppSetPicDescWidth(output_desc_, width_origin);
    acldvppSetPicDescHeight(output_desc_, height_origin);
    acldvppSetPicDescWidthStride(output_desc_, width_stride);
    acldvppSetPicDescHeightStride(output_desc_, height_aligned);
    acldvppSetPicDescSize(output_desc_, buffer_size);

    return TNN_OK;
}

Status AtlasMatConverterAcc::ProcessOutput(Mat& mat) {
    if (DEVICE_ATLAS != mat.GetDeviceType()) {
        // if dst is on host, need to do copy
        LOGD("resize: copy form device to host\n");
        int buffer_size = 0;
        Status tnn_ret  = MatUtils::GetMatByteSize(mat, buffer_size);
        if (TNN_OK != tnn_ret) {
            return tnn_ret;
        }

        aclError acl_ret =
            aclrtMemcpy(mat.GetData(), buffer_size, dvpp_output_buffer_ptr_, buffer_size, ACL_MEMCPY_DEVICE_TO_HOST);
        if (ACL_ERROR_NONE != acl_ret) {
            return Status(TNNERR_ATLAS_RUNTIME_ERROR, "acl memory copy failed");
        }
    }

    return TNN_OK;
}

Status AtlasMatConverterAcc::GetAlignedBufferSize(Mat& mat, int width_align_to, int height_align_to, int& buffer_size,
                                                  int& width_aligned, int& height_aligned) {
    int width      = mat.GetWidth();
    int height     = mat.GetHeight();
    width_aligned  = (width + width_align_to - 1) / width_align_to * width_align_to;
    height_aligned = (height + height_align_to - 1) / height_align_to * height_align_to;

    Mat mat_aligned(mat.GetDeviceType(), mat.GetMatType(),
                    {mat.GetBatch(), mat.GetChannel(), height_aligned, width_aligned}, nullptr);
    Status tnn_ret = MatUtils::GetMatByteSize(mat_aligned, buffer_size);
    if (TNN_OK != tnn_ret) {
        return tnn_ret;
    }

    return TNN_OK;
}

Status AtlasMatConverterAcc::MallocDeviceMemory(void** buffer, int& size, int desired_size) {
    if (nullptr == buffer) {
        return Status(TNNERR_NULL_PARAM, "invalid param");
    }

    aclError acl_ret;

    if (nullptr != *buffer) {
        if (size >= desired_size) {
            return TNN_OK;
        } else {
            // reallocate memory
            LOGD("re allocate memory \n");
            acl_ret = acldvppFree(*buffer);
            if (ACL_ERROR_NONE != acl_ret) {
                LOGE("atlas dvpp free buffer failed!\n");
                return Status(TNNERR_ATLAS_FREE_ERROR, "atlas dvpp free buffer failed");
            }
            *buffer = nullptr;
        }
    }

    acl_ret = acldvppMalloc(buffer, desired_size);
    if (ACL_ERROR_NONE != acl_ret) {
        LOGE("atlas dvpp malloc buffer failed!\n");
        return Status(TNNERR_ATLAS_MALLOC_ERROR, "atlas dvpp malloc buffer failed");
    }
    LOGD("memory addr: 0x%lx   size: %d\n", *buffer, desired_size);
    size = desired_size;

    return TNN_OK;
}

Status AtlasMatConverterAcc::CopyFromHostToDeviceAligned(Mat& src, void* dst, int width_align_to, int height_align_to) {
    Status tnn_ret;
    aclError acl_ret;

    int width_aligned  = 0;
    int height_aligned = 0;
    int buffer_size    = 0;
    tnn_ret = GetAlignedBufferSize(src, width_align_to, height_align_to, buffer_size, width_aligned, height_aligned);
    if (TNN_OK != tnn_ret) {
        return tnn_ret;
    }

    if (width_aligned == src.GetWidth() && height_aligned == src.GetHeight()) {
        // copy directly
        acl_ret = aclrtMemcpy(dst, buffer_size, src.GetData(), buffer_size, ACL_MEMCPY_HOST_TO_DEVICE);
        if (ACL_ERROR_NONE != acl_ret) {
            return Status(TNNERR_ATLAS_RUNTIME_ERROR, "acl memory copy failed");
        }
    } else {
        // need to do padding
        // only support N8UC3 and N8UC4
        MatType mat_type = src.GetMatType();
        if (N8UC3 != mat_type && N8UC4 != mat_type) {
            LOGE("not support this mat type copy from host to device aligned! (mat type: %d)\n", mat_type);
        }

        if (width_aligned == src.GetWidth()) {
            // copy directly
            acl_ret = aclrtMemcpy(dst, buffer_size, src.GetData(), buffer_size, ACL_MEMCPY_HOST_TO_DEVICE);
            if (ACL_ERROR_NONE != acl_ret) {
                return Status(TNNERR_ATLAS_RUNTIME_ERROR, "acl memory copy failed");
            }
        } else {
            // copy form src with stride
            int width_stride = width_aligned;
            if (N8UC3 == mat_type) {
                width_stride *= 3;
            } else if (N8UC4 == mat_type) {
                width_stride *= 4;
            }

            int src_offset = 0;
            int dst_offset = 0;
            for (int h = 0; h < src.GetHeight(); ++h) {
                acl_ret = aclrtMemcpy((char*)dst + dst_offset, src.GetWidth(), (char*)src.GetData() + src_offset,
                                      src.GetWidth(), ACL_MEMCPY_HOST_TO_DEVICE);
                if (ACL_ERROR_NONE != acl_ret) {
                    return Status(TNNERR_ATLAS_RUNTIME_ERROR, "acl memory copy failed");
                }
                dst_offset += width_stride;
                src_offset += src.GetWidth();
            }
        }
    }

    return TNN_OK;
}

int AtlasMatConverterAcc::GetWidthStride(MatType mat_type, int width) {
    if (N8UC3 == mat_type) {
        return width * 3;
    } else if (N8UC4 == mat_type) {
        return width * 4;
    }

    return width;
}

CropParam AtlasMatConverterAcc::ProcessCropParam(CropParam param) {
    CropParam result  = param;
    result.top_left_x = result.top_left_x & (~(0x01));
    result.top_left_y = result.top_left_y & (~(0x01));
    result.width      = (result.width + 1) & (~(0x01));
    result.height     = (result.height + 1) & (~(0x01));
    return result;
}

Status AtlasMatConverterAcc::MatCopyAsync(Mat& dst, Mat& src, int dst_offset, void* stream) {
    if (dst.GetDeviceType() != DEVICE_ATLAS || src.GetDeviceType() != DEVICE_ATLAS) {
        LOGE("MatCopyAsync in AtlasMatConverterAcc only support DEVICE_ATLAS\n");
        return Status(TNNERR_ATLAS_DVPP_NOT_SUPPORT, "MatCopyAsync in AtlasMatConverterAcc only support DEVICE_ATLAS");
    }

    Status tnn_ret = TNN_OK;
    int src_size   = 0;
    int dst_size   = 0;

    tnn_ret = MatUtils::GetMatByteSize(src, src_size);
    if (TNN_OK != tnn_ret) {
        return tnn_ret;
    }
    tnn_ret = MatUtils::GetMatByteSize(dst, dst_size);
    if (TNN_OK != tnn_ret) {
        return tnn_ret;
    }

    if (dst_size < (dst_offset + src_size)) {
        LOGE("invalid offset for MatCopyAsync\n");
        return Status(TNNERR_PARAM_ERR, "invalid offset for MatCopyAsync");
    }

    aclError acl_ret = aclrtMemcpyAsync((char*)dst.GetData() + dst_offset, src_size, src.GetData(), src_size,
                                        ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
    if (ACL_ERROR_NONE != acl_ret) {
        return Status(TNNERR_ATLAS_RUNTIME_ERROR, "acl memory copy failed");
    }

    return TNN_OK;
}

DECLARE_MAT_CONVERTER_CREATER(Atlas);
REGISTER_MAT_CONVERTER(Atlas, DEVICE_ATLAS);

}  // namespace TNN_NS
