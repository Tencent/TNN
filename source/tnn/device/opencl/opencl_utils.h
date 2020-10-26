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

#ifndef TNN_SOURCE_TNN_DEVICE_OPENCL_OPENCL_UTILES_H_
#define TNN_SOURCE_TNN_DEVICE_OPENCL_OPENCL_UTILES_H_

#include <string>
#include <vector>

#include "tnn/device/opencl/opencl_context.h"
#include "tnn/device/opencl/opencl_execute_unit.h"
#include "tnn/device/opencl/opencl_memory.h"
#include "tnn/device/opencl/opencl_runtime.h"

#include "tnn/core/mat.h"
#include "tnn/core/blob.h"
#include "tnn/utils/dims_vector_utils.h"
#include "tnn/interpreter/raw_buffer.h"

namespace TNN_NS {

enum OpenCLBufferFormat {
    CONV2D_FILTER    = 0,
    NHWC_BUFFER      = 1,
    ARGUMENT         = 2,
    DW_CONV2D_FILTER = 3,
    NCHW_BUFFER      = 4,
    NHWC4_BUFFER     = 5,
};

template <typename T, typename Dim>
inline void IOHW2OIHW(const T *src, T *dst, Dim O, Dim I, Dim H, Dim W) {
    for (Dim i = 0; i < I; i++) {
        for (Dim o = 0; o < O; o++) {
            for (Dim h = 0; h < H; h++) {
                for (Dim w = 0; w < W; w++) {
                    dst[o * I * H * W + i * H * W + h * W + w] = src[i * O * H * W + o * H * W + h * W + w];
                }
            }
        }
    }
}

enum GroupWeightsFormat { GOIHW, GIOHW };

template <typename T, typename Dim>
inline void GROUP_PADDING(const T *src, T *dst, Dim G, Dim O, Dim I, Dim H, Dim W, GroupWeightsFormat src_format) {
    int input_channel_per_group = I / G;
    int group_size_in_o         = O / G;

    for (Dim o = 0; o < O; o++) {
        for (Dim i = 0; i < I; i++) {
            for (Dim h = 0; h < H; h++) {
                for (Dim w = 0; w < W; w++) {
                    int dst_idx = o * I * H * W + i * H * W + h * W + w;

                    int group_id  = o / group_size_in_o;
                    int valid_i_b = group_id * input_channel_per_group;
                    int valid_i_e = valid_i_b + input_channel_per_group;
                    if (i < valid_i_b || i >= valid_i_e) {
                        dst[dst_idx] = 0;
                    } else {
                        int g_idx = group_id;
                        int o_idx = o % group_size_in_o;
                        int i_idx = i % input_channel_per_group;
                        int h_idx = h;
                        int w_idx = w;
                        // src is GOIHW
                        int src_idx;
                        if (src_format == GOIHW) {
                            src_idx = g_idx * group_size_in_o * input_channel_per_group * H * W +
                                      o_idx * input_channel_per_group * H * W + i_idx * H * W + h_idx * W + w_idx;
                        } else {
                            // src is GIOHW
                            src_idx = g_idx * input_channel_per_group * group_size_in_o * H * W +
                                      i_idx * group_size_in_o * H * W + o_idx * H * W + h_idx * W + w_idx;
                        }
                        dst[dst_idx] = src[src_idx];
                    }
                }
            }
        }
    }
}

inline cl::Buffer &GetOpenCLBuffer(const OpenCLMemory *blob) {
    return (*(cl::Buffer *)(blob->GetData()));
}
inline cl::Image &GetOpenCLImage(const OpenCLMemory *blob) {
    return (*(cl::Image *)(blob->GetData()));
}

std::vector<int> GetImageShape(const OpenCLMemory *image);

void GetProfilingTime(const cl::Event *event, double &kernel_time, double &event_queued, double &event_submit,
                      double &event_start, double &event_end);

Status RunKernel(const cl::Kernel &kernel, const std::vector<uint32_t> &gws, const std::vector<uint32_t> &lws,
                 cl::CommandQueue *command_queue, std::string name = "", OpenCLProfilingData *pdata = nullptr);

std::vector<uint32_t> AdrenoLocalSize2D(const std::vector<uint32_t> &gws, const GpuInfo gpu_info,
                                        const uint32_t compute_units, const uint32_t max_workgroup_size,
                                        const uint32_t subgroup_size);

Status AdjustBuildOptionForFp32(std::set<std::string>& build_options);

std::vector<uint32_t> LocalWS3DDefault(OpenCLExecuteUnit &unit);

std::vector<uint32_t> LocalWS3DDefault(const std::vector<uint32_t> &gws, const uint32_t max_workgroup_size,
                                       const uint32_t subgroup_size = 0);

std::vector<uint32_t> LocalWS2DDefault(OpenCLExecuteUnit &unit);

std::vector<uint32_t> LocalWS2DDefault(const std::vector<uint32_t> &gws, const uint32_t max_workgroup_size,
                                       const uint32_t subgroup_size = 0);

Status CopyBufferToImage(OpenCLRuntime *runtime, OpenCLContext *context, const cl::Buffer &buffer,
                         const cl::Image &image, int w, int h, bool need_wait = false);

Status CopyImageToImage(OpenCLRuntime *runtime, OpenCLContext *context, const cl::Image &src, const cl::Image &dst,
                        int w, int h, bool need_wait = false, OpenCLProfilingData *pdata = nullptr);

Status CopyBufferToMat(Mat &mat, cl::Buffer& buffer, DimsVector& dims, const int buffer_size,
                       const MatType& mat_type, cl::CommandQueue *command_queue);

Status CopyMatToBuffer(Mat &mat, cl::Buffer& buffer, DimsVector& dims, const int buffer_size,
                       const MatType& mat_type, cl::CommandQueue *command_queue);

uint32_t gcd(uint32_t number1, uint32_t number2);

Status CreateExecuteUnit(OpenCLExecuteUnit &unit, const std::string &program_name, const std::string &kernel_name,
                         const std::set<std::string> &build_opt = {});

uint32_t SetExecuteUnit3DSizeInfoDefault(OpenCLExecuteUnit &unit, DimsVector dims);

uint32_t SetExecuteUnit2DSizeInfoDefault(OpenCLExecuteUnit &unit, DimsVector dims);

}  // namespace TNN_NS
#endif  // TNN_SOURCE_TNN_DEVICE_OPENCL_OPENCL_UTILES_H_
