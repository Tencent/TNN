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

#include "tnn/device/cpu/cpu_mat_converter.h"

#include <algorithm>
#include <cstring>

#include "tnn/core/blob_int8.h"
#include "tnn/core/macro.h"
#include "tnn/utils/naive_compute.h"
#include "tnn/utils/bfp16.h"
#include "tnn/utils/bfp16_utils.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {
    
Status CpuMatConverterAcc::Copy(Mat& src, Mat& dst, void* command_queue) {
    Status ret            = TNN_OK;
    return ret; 
}

Status CpuMatConverterAcc::Resize(Mat& src, Mat& dst, ResizeParam param, void* command_queue) {
    Status ret            = TNN_OK;
    std::vector<int> src_dims = src.GetDims();
    std::vector<int> dst_dims = dst.GetDims();
    int w = dst.GetWidth();
    int h = dst.GetHeight();
    int rows = src.GetWidth();
    int cols = src.GetHeight();
    int channel = src.GetChannel();
    unsigned char* ptr = (unsigned char*)src.GetData();
    unsigned char* part = (unsigned char*)malloc(sizeof(unsigned char)*w*rows*channel);
    unsigned char* dst_ptr = (unsigned char*)dst.GetData();
    float w_scale = param.scale_w;
    float h_scale = param.scale_h;
    float wscale = cols / (float)w;
    float hscale = rows / (float)h;

	unsigned char val;

    if (src.GetMatType() == NCHW_FLOAT) {
        ret = Status(TNNERR_PARAM_ERR, "convert type not support yet");
    } else if (src.GetMatType() == N8UC4) {
        ret = Status(TNNERR_PARAM_ERR, "convert type not support yet");
    } else if (src.GetMatType() == N8UC3) {
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < w; c++)
                for (int k = 0; k < channel; k++)
                {
                    if (c == 0 || c == w - 1)
                    {
                        val = ptr[r*cols*channel + c*channel + k];
                    }
                    else 
                    {
                        float x1 = c * wscale;
                        int x2 = (int)x1;
                        float diffx = x1 - x2;
                        val = (1 - diffx)*ptr[r*cols*channel + c*channel + x2] + diffx*ptr[r*cols*channel + c*channel + x2 + 1];
                    }
                    part[r*w*channel + c*channel + k] = val;
                }

        for (int r = 0; r < h; r++)
        {
            float y1 = r * hscale;
            int y2 = (int)y1;
            float diffy = y1 - y2;
            for (int c = 0; c < w; c++)
                for (int k = 0; k < channel; k++)
                {
                    if (r == 0 || r == h - 1)
                    {
                        val = ptr[r*w*channel + c*channel+k];
                    }
                    else
                    {
                        val = (1 - diffy)* ptr[y2*w*channel + c*channel + k] + diffy * ptr[(y2+1)*w*channel + c*channel + k];
                    }
                    dst_ptr[r*w*channel + c*channel + k] = val;
                }
        }
    free(part);
    } else if (src.GetMatType() == NGRAY) {
        ret = Status(TNNERR_PARAM_ERR, "convert type not support yet");
    } else if (src.GetMatType() == RESERVED_BFP16_TEST) {
        ret = Status(TNNERR_PARAM_ERR, "convert type not support yet");
    } else {
        ret = Status(TNNERR_PARAM_ERR, "convert type not support yet");
    }
    return ret;
}

Status CpuMatConverterAcc::Crop(Mat& src, Mat& dst, CropParam param, void* command_queue) {
    Status ret            = TNN_OK;
    if (src.GetData() == nullptr) {
        return Status(TNNERR_NULL_PARAM, "input mat is null");
    }

    if (src.GetDeviceType() != dst.GetDeviceType()) {
        return Status(TNNERR_PARAM_ERR, "src and dst mat type must be same");
    }

    if (dst.GetData() == nullptr) {
        dst = Mat(dst.GetDeviceType(), dst.GetMatType(), dst.GetDims());
    }

    if (src.GetMatType() == NGRAY) {
        // element size 1
        auto src_ptr = GET_OFFSET_PTR(src.GetData(), param.top_left_x + param.top_left_y * src.GetWidth());
        auto dst_ptr = GET_OFFSET_PTR(dst.GetData(), 0);
        mat_memcpy_2d(src_ptr, dst_ptr, param.width, param.height, src.GetWidth(), dst.GetWidth());
    } else if (src.GetMatType() == N8UC3) {
        // element size 3
        auto src_ptr = GET_OFFSET_PTR(src.GetData(), (param.top_left_x + param.top_left_y * src.GetWidth()) * 3);
        auto dst_ptr = GET_OFFSET_PTR(dst.GetData(), 0);
        mat_memcpy_2d(src_ptr, dst_ptr, param.width * 3, param.height, src.GetWidth() * 3, dst.GetWidth() * 3);
    } else if (src.GetMatType() == N8UC4) {
        // element size 4
        auto src_ptr = GET_OFFSET_PTR(src.GetData(), (param.top_left_x + param.top_left_y * src.GetWidth()) * 4);
        auto dst_ptr = GET_OFFSET_PTR(dst.GetData(), 0);
        mat_memcpy_2d(src_ptr, dst_ptr, param.width * 4, param.height, src.GetWidth() * 4, dst.GetWidth() * 4);
    } else if (src.GetMatType() == NNV21 || src.GetMatType() == NNV12) {
        if (param.top_left_x % 2 || param.top_left_y % 2 || param.width % 2 || param.height % 2) {
            return Status(TNNERR_PARAM_ERR, "corp param can not be odd");
        }
        // crop y
        auto src_ptr = GET_OFFSET_PTR(src.GetData(), param.top_left_x + param.top_left_y * param.width);
        auto dst_ptr = GET_OFFSET_PTR(dst.GetData(), 0);
        mat_memcpy_2d(src_ptr, dst_ptr, param.width, param.height, src.GetWidth(), dst.GetWidth());
        // crop uv
        src_ptr = GET_OFFSET_PTR(
            src.GetData(), src.GetWidth() * src.GetHeight() + param.top_left_x + param.top_left_y * src.GetWidth() / 2);
        dst_ptr = GET_OFFSET_PTR(dst.GetData(), dst.GetWidth() * dst.GetHeight());
        mat_memcpy_2d(src_ptr, dst_ptr, param.width, param.height / 2, src.GetWidth(), dst.GetWidth());
    } else {
        return Status(TNNERR_PARAM_ERR, "convert type not support yet");
    }

    return ret;
}

Status CpuMatConverterAcc::WarpAffine(Mat& src, Mat& dst, WarpAffineParam param, void* command_queue) {
    Status ret            = TNN_OK;
    return ret;
    // auto cl_command_queue = static_cast<cl::CommandQueue *>(command_queue);
    // if (cl_command_queue == nullptr) {
    //     LOGE("Get OpenCL command queue failed!\n");
    //     return Status(TNNERR_NULL_PARAM, "Get OpenCL command queue failed!");
    // } 
    // if(execute_map_.count("WarpAffine") == 0) {        

    // }
}

void CpuMatConverterAcc::mat_memcpy_2d(void* src, void* dst, int width, int height, int src_stride, int dst_stride) {
    auto src_ptr = reinterpret_cast<uint8_t*>(src);
    auto dst_ptr = reinterpret_cast<uint8_t*>(dst);

    for (int h = 0; h < height; h++) {
        memcpy(dst_ptr, src_ptr, width);
        src_ptr += src_stride;
        dst_ptr += dst_stride;
    }

}
// Status CpuBlobConverterAcc::ConvertToMat(Mat &image, MatConvertParam param, void *command_queue) {
//     return ConvertToMatAsync(image, param, command_queue);
// }

// Status CpuBlobConverterAcc::ConvertFromMat(Mat &image, MatConvertParam param, void *command_queue) {
//     return ConvertFromMatAsync(image, param, command_queue);
// }

// Status CpuBlobConverterAcc::ConvertNCHWToNHWC(uint8_t *src, uint8_t *dst) {
//     return TNN_OK;
// }

// Status CpuBlobConverterAcc::ConvertNHWCToNCHW(uint8_t *src, uint8_t *dst) {
//     return TNN_OK;
// }

DECLARE_MAT_CONVERTER_CREATER(Cpu);
REGISTER_MAT_CONVERTER(Cpu, DEVICE_NAIVE);

}  // namespace TNN_NS

// unsigned char getval(unsigned char* ptr, int cols, int height, int width)
// {
// 	return ptr[height*cols + width];
// }

// void setval(unsigned char* ptr, int cols, int height, int width, unsigned char pixel)
// {
// 	ptr[height*cols + width] = pixel;
// }

// unsigned char* resize(unsigned char* ptr, int rows, int cols, int h, int w)
// {
// 	unsigned char* part = (unsigned char*)malloc(sizeof(unsigned char)*w*rows);
// 	unsigned char* dst = (unsigned char*)malloc(sizeof(unsigned char)*w*h);

// 	float wscale = cols / (float)w;
// 	float hscale = rows / (float)h;

// 	unsigned char val;
// 	for (int r = 0; r < rows; r++)
// 		for (int c = 0; c < w; c++)
// 		{
// 			if (c == 0 || c == w - 1)
// 			{
// 				val = getval(ptr, cols, r, c);
// 			}
// 			else
// 			{
// 				float x1 = c * wscale;
// 				int x2 = (int)x1;
// 				float diffx = x1 - x2;

// 				val = (1 - diffx)*getval(ptr, cols, r, x2) + diffx*getval(ptr, cols, r, x2 + 1);
// 			}
// 			setval(part, w, r, c, val);
// 		}

// 		for (int r = 0; r < h; r++)
// 		{
// 			float y1 = r * hscale;
// 			int y2 = (int)y1;
// 			float diffy = y1 - y2;
// 			for (int c = 0; c < w; c++)
// 			{
// 				if (r == 0 || r == h - 1)
// 				{
// 					val = getval(part, w, r, c);
// 				}
// 				else
// 				{
// 					val = (1 - diffy)*getval(part, w, y2, c) + diffy * getval(part, w, y2+1, c);
// 				}
// 				setval(dst, w, r, c, val);
// 			}
// 		}
// 	free(part);
// 	return dst;
// }

//     uchar* dataDst = dst.data;
//     int stepDst = matDst1.step;
//     uchar* dataSrc = src.data;
//     int stepSrc = matSrc.step;
//     int iWidthSrc = matSrc.cols;
//     int iHiehgtSrc = matSrc.rows;

//     for (int j = 0; j < matDst1.rows; ++j)
//     {
//         float fy = (float)((j + 0.5) * scale_y - 0.5);
//         int sy = cvFloor(fy);
//         fy -= sy;
//         sy = std::min(sy, iHiehgtSrc - 2);
//         sy = std::max(0, sy);

//         short cbufy[2];
//         cbufy[0] = cv::saturate_cast<short>((1.f - fy) * 2048);
//         cbufy[1] = 2048 - cbufy[0];

//         for (int i = 0; i < matDst1.cols; ++i)
//         {
//             float fx = (float)((i + 0.5) * scale_x - 0.5);
//             int sx = cvFloor(fx);
//             fx -= sx;

//             if (sx < 0) {
//                 fx = 0, sx = 0;
//             }
//             if (sx >= iWidthSrc - 1) {
//                 fx = 0, sx = iWidthSrc - 2;
//             }

//             short cbufx[2];
//             cbufx[0] = cv::saturate_cast<short>((1.f - fx) * 2048);
//             cbufx[1] = 2048 - cbufx[0];

//             for (int k = 0; k < matSrc.channels(); ++k)
//             {
//                 *(dataDst+ j*stepDst + 3*i + k) = (*(dataSrc + sy*stepSrc + 3*sx + k) * cbufx[0] * cbufy[0] + 
//                     *(dataSrc + (sy+1)*stepSrc + 3*sx + k) * cbufx[0] * cbufy[1] + 
//                     *(dataSrc + sy*stepSrc + 3*(sx+1) + k) * cbufx[1] * cbufy[0] + 
//                     *(dataSrc + (sy+1)*stepSrc + 3*(sx+1) + k) * cbufx[1] * cbufy[1]) >> 22;
//             }
//         }
//     }
//     cv::imwrite("linear_1.jpg", matDst1);

//     cv::resize(matSrc, matDst2, matDst1.size(), 0, 0, 1);
//     cv::imwrite("linear_2.jpg", matDst2);

// namespace TNN_NS {

// Status CpuMatConverterAcc::Resize(Mat& src, Mat& dst, ResizeParam param, void* command_queue = NULL) {
//     Status ret            = TNN_OK;
//     auto cl_command_queue = static_cast<cl::CommandQueue *>(command_queue);
//     if (cl_command_queue == nullptr) {
//         LOGE("Get OpenCL command queue failed!\n");
//         return Status(TNNERR_NULL_PARAM, "Get OpenCL command queue failed!");
//     }
//     const std::string key = "Resize";
//     OpenCLExecuteUnit unit;
//     if(execute_map_.count(key) == 0) {
//         std::string program_name = "upsample";
//         std::string kernel_name = "Bilinear";
//         ret = CreateExecuteUnit(unit, program_name, kernel_name);
//         if(ret != TNN_OK) {
//             return ret;
//         }
//         execute_map_[key] = unit; 
//     }
// }

// DECLARE_MAT_CONVERTER_CREATER(OpenCL);
// REGISTER_MAT_CONVERTER(OpenCL, DEVICE_OPENCL);

// }  // namespace TNN_NS

