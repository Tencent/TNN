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

    int dst_width = src.GetWidth() * param.scale_w;
    int dst_height = src.GetHeight() * param.scale_h;
    bool need_do_paste = false;
    if (dst_width > dst.GetWidth() || dst_height > dst.GetHeight()) {
        LOGE("resize param is invailed! (need %dx%d  but dst memory %dx%d)\n", dst_width, dst_height, dst.GetWidth(), dst.GetHeight());
        return Status(TNNERR_NULL_PARAM, "resize param is invalid!");
    }

    need_do_paste = dst_width != dst.GetWidth() || dst_height != dst.GetHeight();

    Status ret = TNN_OK;
    aclError acl_ret = ACL_ERROR_NONE;

    ret = PrepareInput(src);
    if (TNN_OK != ret) {
        return ret;
    }

    ret = PrepareOutput(dst);
    if (TNN_OK != ret) {
        return ret;
    }

    if (need_do_paste) {
        acldvppRoiConfig* crop_roi_config = acldvppCreateRoiConfig(0, (src.GetWidth() & (~0x01)) - 1, 0, (src.GetHeight() & (~0x01)) - 1);
        if (nullptr == crop_roi_config) {
            LOGE("create crop roi config in resize failed\n");
            return Status(TNNERR_ATLAS_RUNTIME_ERROR, "acldvppCreateRoiConfig failed");
        }
        acldvppRoiConfig* paste_roi_config = acldvppCreateRoiConfig(0, (dst_width & (~0x01)) - 1, 0, (dst_height & (~0x01)) - 1);
        if (nullptr == paste_roi_config) {
            LOGE("create crop roi config in resize failed\n");
            return Status(TNNERR_ATLAS_RUNTIME_ERROR, "acldvppCreateRoiConfig failed");
        }

        acl_ret = acldvppVpcCropAndPasteAsync(dvpp_channel_desc_, input_desc_, output_desc_, crop_roi_config, paste_roi_config, atlas_cmd_queue->stream);
        if (ACL_ERROR_NONE != acl_ret) {
            LOGE("acldvppVpcCropAndPasteAsync failed, ret = %d\n", acl_ret);
            return Status(TNNERR_ATLAS_RUNTIME_ERROR, "acldvppVpcCropAndPasteAsync failed");
        }

        if (nullptr != crop_roi_config) {
            acl_ret = acldvppDestroyRoiConfig(crop_roi_config);
            if (ACL_ERROR_NONE != acl_ret) {
                LOGE("acldvppDestroyRoiConfig failed, ret = %d\n", acl_ret);
                return Status(TNNERR_ATLAS_RUNTIME_ERROR, "acldvppDestroyRoiConfig failed");
            }
        }
        if (nullptr != paste_roi_config) {
            acl_ret = acldvppDestroyRoiConfig(paste_roi_config);
            if (ACL_ERROR_NONE != acl_ret) {
                LOGE("acldvppDestroyRoiConfig failed, ret = %d\n", acl_ret);
                return Status(TNNERR_ATLAS_RUNTIME_ERROR, "acldvppDestroyRoiConfig failed");
            }
        }
    } else {
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
            ret = acldvppDestroyResizeConfig(resize_config);
            if (ACL_ERROR_NONE != ret) {
                LOGE("acldvppDestroyResizeConfig failed, ret = %d\n", ret);
            }
            resize_config = nullptr;
        }
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
              {dst.GetBatch(), dst.GetChannel(), param_real.height, param_real.width}, nullptr);
    ret                  = PrepareOutput(dst);
    if (TNN_OK != ret) {
        return ret;
    }

    acldvppRoiConfig* crop_roi_config = acldvppCreateRoiConfig(param_real.top_left_x, param_real.top_left_x + param_real.width - 1,param_real.top_left_y, param_real.top_left_y + param_real.height - 1);
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

    return TNN_OK;
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

Status AtlasMatConverterAcc::PrepareOutput(Mat& mat) {
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

    int width_origin = mat.GetWidth();
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
        mat = Mat(device_type, mat.GetMatType(), {mat.GetBatch(), mat.GetChannel(), height_origin, width_origin});
    }

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
        return CopyFromDeviceToHostAligned(dvpp_output_buffer_ptr_, mat, 16, 2);
    }

    return TNN_OK;
}

Status AtlasMatConverterAcc::GetAlignedBufferSize(Mat& mat, int width_align_to, int height_align_to, int& buffer_size,
                                                  int& width_aligned, int& height_aligned) {
    int width      = mat.GetWidth();
    int height     = mat.GetHeight();
    width_aligned  = (width + width_align_to - 1) / width_align_to * width_align_to;
    height_aligned = (height + height_align_to - 1) / height_align_to * height_align_to;

    // calculate output_buffer_size
    MatType mat_type = mat.GetMatType();
    if (N8UC3 == mat_type) {
        buffer_size = width_aligned * height_aligned * 3;
    } else if (N8UC4 == mat_type) {
        buffer_size = width_aligned * height_aligned * 4;
    } else if (NNV12 == mat_type || NNV21 == mat_type) {
        buffer_size = width_aligned * height_aligned * 3 / 2;
    } else {
        LOGE("atlas mat resize not support this input mat type (mat type is %d)!\n", mat_type);
        return Status(TNNERR_ATLAS_DVPP_NOT_SUPPORT, "atlas mat resize not support this input mat type");
    }

    if (0 == buffer_size) {
        LOGE("atlas mat resize invalid input buffer size!\n");
        return Status(TNNERR_ATLAS_DVPP_NOT_SUPPORT, "atlas mat resize invalid input buffer size");
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

Status AtlasMatConverterAcc::CopyFromDeviceToHostAligned(void* src, Mat& dst, int width_align_to, int height_align_to) {
    Status tnn_ret;
    aclError acl_ret;

    int width_aligned  = 0;
    int height_aligned = 0;
    int buffer_size    = 0;
    tnn_ret = GetAlignedBufferSize(dst, width_align_to, height_align_to, buffer_size, width_aligned, height_aligned);
    if (TNN_OK != tnn_ret) {
        return tnn_ret;
    }

    if (buffer_size > GetMatByteSize(dst)) {
        LOGE("invalid buffer size to copy from device to host\n");
        return Status(TNNERR_PARAM_ERR, "invalid buffer size to copy from device to host");
    }

    // copy directly
    acl_ret = aclrtMemcpy(dst.GetData(), buffer_size, src, buffer_size, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ACL_ERROR_NONE != acl_ret) {
        return Status(TNNERR_ATLAS_RUNTIME_ERROR, "acl memory copy failed");
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

int AtlasMatConverterAcc::GetMatByteSize(Mat& mat) {
    int type_size = 1;
    if (NCHW_FLOAT == mat.GetMatType()) {
        type_size *= 4;
    } else if (N8UC3 == mat.GetMatType()) {
    } else if (N8UC4 == mat.GetMatType()) {
    } else if (NGRAY == mat.GetMatType()) {
    } else if (NNV21 == mat.GetMatType()) {
    } else if (NNV12 == mat.GetMatType()) {
    } else {
        LOGE("invalid mat type(%d) to get mat size\n", mat.GetMatType());
        return 0;
    }

    int dim_size = DimsVectorUtils::Count(mat.GetDims());
    return dim_size * type_size;
}

CropParam AtlasMatConverterAcc::ProcessCropParam(CropParam param) {
    CropParam result  = param;
    result.top_left_x = result.top_left_x & (~(0x01));
    result.top_left_y = result.top_left_y & (~(0x01));
    result.width      = (result.width + 1) & (~(0x01));
    result.height     = (result.height + 1) & (~(0x01));
    return result;
}

DECLARE_MAT_CONVERTER_CREATER(Atlas);
REGISTER_MAT_CONVERTER(Atlas, DEVICE_ATLAS);

}  // namespace TNN_NS
