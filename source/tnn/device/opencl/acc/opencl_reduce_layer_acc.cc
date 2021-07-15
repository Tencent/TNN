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

#include "tnn/device/opencl/acc/opencl_reduce_layer_acc.h"
#include "tnn/device/opencl/imagebuffer_convertor.h"

namespace TNN_NS {

#define LowOpParallelismThre 256
#define HighOpIntensityThre 128

DimsVector PadDims(DimsVector dims, std::vector<int> axis) {
    std::sort(axis.begin(), axis.end());
    for (const auto item : axis) {
        dims.insert(dims.begin() + item, 1);
    }

    return dims;
}

Status OpenCLReduceLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                  const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init Reduce Acc\n");
    Status ret = OpenCLLayerAcc::Init(context, param, resource, inputs, outputs);
    CHECK_TNN_OK(ret)

    auto reduce_param = dynamic_cast<ReduceLayerParam *>(param);
    if (!reduce_param) {
        LOGE("Error: layer param is null\n");
        return Status(TNNERR_MODEL_ERR, "Error: layer param is null");
    }

    auto output_dims = outputs[0]->GetBlobDesc().dims;

    int hb   = DimsFunctionUtils::GetDim(output_dims, 0) * DimsFunctionUtils::GetDim(output_dims, 2);
    int cw   = DimsFunctionUtils::GetDim(output_dims, 3) * UP_DIV(DimsFunctionUtils::GetDim(output_dims, 1), 4);

    auto input_dims  = inputs[0]->GetBlobDesc().dims;

    if (reduce_param->keep_dims == 0) {
        execute_units_.resize(3);
    }

    if (reduce_param->axis.size() == 1) {
        int axis = reduce_param->axis[0];
        axis     = axis >= 0 ? axis : axis + (int)input_dims.size();

        int axis_n = DimsFunctionUtils::GetDim(input_dims, axis);

        run_local_work_ = cw * hb < LowOpParallelismThre && axis_n >= HighOpIntensityThre;

        run_3d_ndrange_         = false;
        std::string kernel_name;
        if (axis == 0) {
            kernel_name = "ReduceC0";
        } else if (axis == 1) {
            kernel_name = "ReduceC1";
        } else if (axis == 2) {
            kernel_name = "ReduceC2";
        } else {
            kernel_name = "ReduceC3";
        }

        if (run_local_work_) {
            kernel_name += "Local";
        }

        std::set<std::string> build_options = CreateBuildOptions();

        ret = CreateExecuteUnit(execute_units_[0], "reduce", kernel_name, build_options);
        if (ret != TNN_OK) {
            LOGE("create execute unit failed!\n");
            return ret;
        }
    } else {
        run_3d_ndrange_         = false;
        std::string kernel_name = "ReduceMultiAxis";

        std::set<std::string> build_options = CreateBuildOptions();

        ret = CreateExecuteUnit(execute_units_[0], "reduce", kernel_name, build_options);
        if (ret != TNN_OK) {
            LOGE("create execute unit failed!\n");
            return ret;
        }
    }

    if (reduce_param->keep_dims == 0) {
        // image->buffer
        {
            ret = CreateExecuteUnit(execute_units_[1], "image_to_buffer", "ImageToNCHWBuffer");
            if (ret != TNN_OK) {
                LOGE("create execute unit failed!\n");
                return ret;
            }
        }

        // buffer->image
        {
            ret = CreateExecuteUnit(execute_units_[2], "buffer_to_image", "NCHWBufferToImage");
            if (ret != TNN_OK) {
                LOGE("create execute unit failed!\n");
                return ret;
            }
        }
    }

    return TNN_OK;
}

OpenCLReduceLayerAcc::~OpenCLReduceLayerAcc() {}

Status OpenCLReduceLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Reduce Layer Reshape\n");
    Status ret = OpenCLLayerAcc::Reshape(inputs, outputs);
    CHECK_TNN_OK(ret)

    auto reduce_param = dynamic_cast<ReduceLayerParam *>(param_);
    if (!reduce_param) {
        LOGE("Error: layer param is null\n");
        return Status(TNNERR_MODEL_ERR, "Error: layer param is null");
    }

    ASSERT(inputs.size() == 1);

    auto input_dims  = inputs[0]->GetBlobDesc().dims;
    auto output_dims = outputs[0]->GetBlobDesc().dims;

    const int keep_dims    = reduce_param->keep_dims;
    auto final_output_dims = output_dims;
    if (keep_dims == 0) {
        output_dims = PadDims(output_dims, reduce_param->axis);
        GenerateTempImage(output_dims);
    }

    auto* reduce_output_ptr = keep_dims != 0 ? (cl::Image *)outputs[0]->GetHandle().base : (cl::Image *)inter_image_->GetData();

    int hb   = DimsFunctionUtils::GetDim(output_dims, 0) * DimsFunctionUtils::GetDim(output_dims, 2);
    int cw   = DimsFunctionUtils::GetDim(output_dims, 3) * UP_DIV(DimsFunctionUtils::GetDim(output_dims, 1), 4);
    int c4_n = DimsFunctionUtils::GetDim(input_dims, 1) / 4;
    int c4_r = DimsFunctionUtils::GetDim(input_dims, 1) % 4;
    int cw4  = DimsFunctionUtils::GetDim(input_dims, 3) * c4_n;

    if (reduce_param->axis.size() == 1) {
        int axis = reduce_param->axis[0];
        axis     = axis >= 0 ? axis : axis + (int)input_dims.size();

        int axis_n = DimsFunctionUtils::GetDim(input_dims, axis);

        auto &unit            = execute_units_[0];
        uint32_t workgroup_size = 0;

        OpenCLRuntime *opencl_runtime = OpenCLRuntime::GetInstance();
        int type_size = sizeof(float);
        if (opencl_runtime->GetPrecision() != PRECISION_HIGH) {
            type_size = 2;
        }

        if (run_local_work_) {
            workgroup_size = std::min(static_cast<uint32_t>(unit.local_mem_size / (4 * type_size)),
                                    unit.workgroupsize_max);
            workgroup_size = std::min(static_cast<uint32_t>(axis == 1 ? c4_n : axis_n), workgroup_size);
            int temp_size = 1;
            while ((temp_size <<= 1) <= workgroup_size);
            workgroup_size = temp_size >> 1;

            unit.global_work_size = {static_cast<uint32_t>(cw * workgroup_size), static_cast<uint32_t>(hb)};
            unit.local_work_size  = {workgroup_size, 1};
        } else {
            unit.global_work_size = {static_cast<uint32_t>(cw), static_cast<uint32_t>(hb)};
            unit.local_work_size  = LocalWS2DDefault(unit);
        }

        uint32_t idx = 0;
        execute_units_[0].ocl_kernel.setArg(idx++, execute_units_[0].global_work_size[0]);
        execute_units_[0].ocl_kernel.setArg(idx++, execute_units_[0].global_work_size[1]);

        execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)inputs[0]->GetHandle().base));
        execute_units_[0].ocl_kernel.setArg(idx++, *reduce_output_ptr);
        execute_units_[0].ocl_kernel.setArg(idx++, DimsFunctionUtils::GetDim(input_dims, 0));
        execute_units_[0].ocl_kernel.setArg(idx++, DimsFunctionUtils::GetDim(input_dims, 1));
        execute_units_[0].ocl_kernel.setArg(idx++, DimsFunctionUtils::GetDim(input_dims, 2));
        execute_units_[0].ocl_kernel.setArg(idx++, DimsFunctionUtils::GetDim(input_dims, 3));
        execute_units_[0].ocl_kernel.setArg(idx++, c4_n);
        execute_units_[0].ocl_kernel.setArg(idx++, c4_r);
        execute_units_[0].ocl_kernel.setArg(idx++, cw4);
        execute_units_[0].ocl_kernel.setArg(idx++, axis_n);

        if (run_local_work_) {
            if (axis == 1) {
                execute_units_[0].ocl_kernel.setArg(idx++, UP_DIV(c4_n, workgroup_size));
            } else {
                execute_units_[0].ocl_kernel.setArg(idx++, UP_DIV(axis_n, workgroup_size));
            }
            execute_units_[0].ocl_kernel.setArg(idx++, workgroup_size * 4 * type_size, nullptr);
        }
    } else {
        auto &unit              = execute_units_[0];
        uint32_t workgroup_size = 0;

        int axis_n = 1;
        std::vector<int> axis_nhwc = {0, 0, 0, 0};
        for (int i = 0; i < reduce_param->axis.size(); i++) {
            int axis = reduce_param->axis[i];
            axis     = axis >= 0 ? axis : axis + (int)input_dims.size();
            switch(axis) {
                case 0:
                    if (!axis_nhwc[0]) {
                        axis_n *= DimsFunctionUtils::GetDim(input_dims, axis);
                        axis_nhwc[0] = 1;
                    }
                    break;
                case 1:
                    if (!axis_nhwc[3]) {
                        axis_n *= DimsFunctionUtils::GetDim(input_dims, axis);
                        axis_nhwc[3] = 1;
                    }
                    break;
                case 2:
                    if (!axis_nhwc[1]) {
                        axis_n *= DimsFunctionUtils::GetDim(input_dims, axis);
                        axis_nhwc[1] = 1;
                    }
                    break;
                case 3:
                    if (!axis_nhwc[2]) {
                        axis_n *= DimsFunctionUtils::GetDim(input_dims, axis);
                        axis_nhwc[2] = 1;
                    }
                    break;
            }
        }

        unit.global_work_size = {static_cast<uint32_t>(cw), static_cast<uint32_t>(hb)};
        unit.local_work_size  = LocalWS2DDefault(unit);

        uint32_t idx = 0;
        execute_units_[0].ocl_kernel.setArg(idx++, execute_units_[0].global_work_size[0]);
        execute_units_[0].ocl_kernel.setArg(idx++, execute_units_[0].global_work_size[1]);

        execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)inputs[0]->GetHandle().base));
        execute_units_[0].ocl_kernel.setArg(idx++, *reduce_output_ptr);
        execute_units_[0].ocl_kernel.setArg(idx++, DimsFunctionUtils::GetDim(input_dims, 0));
        execute_units_[0].ocl_kernel.setArg(idx++, DimsFunctionUtils::GetDim(input_dims, 1));
        execute_units_[0].ocl_kernel.setArg(idx++, DimsFunctionUtils::GetDim(input_dims, 2));
        execute_units_[0].ocl_kernel.setArg(idx++, DimsFunctionUtils::GetDim(input_dims, 3));
        execute_units_[0].ocl_kernel.setArg(idx++, c4_n);
        execute_units_[0].ocl_kernel.setArg(idx++, c4_r);
        execute_units_[0].ocl_kernel.setArg(idx++, cw4);
        execute_units_[0].ocl_kernel.setArg(idx++, axis_n);
        execute_units_[0].ocl_kernel.setArg(idx++, 4 * sizeof(int), axis_nhwc.data());
    }

    if (keep_dims == 0) {
        OpenCLRuntime *opencl_runtime = OpenCLRuntime::GetInstance();

        int size0 = UP_DIV(DimsFunctionUtils::GetDim(final_output_dims, 1), 4) * 4 *
                    DimsFunctionUtils::GetDim(final_output_dims, 0) * DimsFunctionUtils::GetDim(final_output_dims, 2) *
                    DimsFunctionUtils::GetDim(final_output_dims, 3);
        int size1 = UP_DIV(DimsFunctionUtils::GetDim(output_dims, 1), 4) * 4 *
                    DimsFunctionUtils::GetDim(output_dims, 0) * DimsFunctionUtils::GetDim(output_dims, 2) *
                    DimsFunctionUtils::GetDim(output_dims, 3);
        int blob_size = std::max(size0, size1) * sizeof(float);

        inter_buffer_ = std::make_shared<cl::Buffer>(*opencl_runtime->Context(), CL_MEM_READ_WRITE, blob_size);

        // image->buffer
        {
            uint32_t idx = SetExecuteUnit2DSizeInfoDefault(execute_units_[1], output_dims);
            execute_units_[1].ocl_kernel.setArg(idx++, *inter_buffer_.get());
            execute_units_[1].ocl_kernel.setArg(idx++,
                                                static_cast<uint32_t>(DimsFunctionUtils::GetDim(output_dims, 2)));
            execute_units_[1].ocl_kernel.setArg(idx++,
                                                static_cast<uint32_t>(DimsFunctionUtils::GetDim(output_dims, 3)));
            execute_units_[1].ocl_kernel.setArg(idx++,
                                                static_cast<uint32_t>(DimsFunctionUtils::GetDim(output_dims, 1)));
            execute_units_[1].ocl_kernel.setArg(idx++, *((cl::Image *)inter_image_->GetData()));
        }

        // buffer->image
        {
            uint32_t idx = SetExecuteUnit2DSizeInfoDefault(execute_units_[2], final_output_dims);
            execute_units_[2].ocl_kernel.setArg(idx++, *inter_buffer_.get());
            execute_units_[2].ocl_kernel.setArg(idx++,
                                                static_cast<uint32_t>(DimsFunctionUtils::GetDim(final_output_dims, 2)));
            execute_units_[2].ocl_kernel.setArg(idx++,
                                                static_cast<uint32_t>(DimsFunctionUtils::GetDim(final_output_dims, 3)));
            execute_units_[2].ocl_kernel.setArg(idx++,
                                                static_cast<uint32_t>(DimsFunctionUtils::GetDim(final_output_dims, 1)));
            execute_units_[2].ocl_kernel.setArg(idx++, *((cl::Image *)outputs[0]->GetHandle().base));
        }
    }

    return TNN_OK;
}

Status OpenCLReduceLayerAcc::GenerateTempImage(DimsVector dims) {
    OpenCLRuntime *opencl_runtime = OpenCLRuntime::GetInstance();

    // copy param data into clBuffer
    shared_ptr<OpenCLMemory> buffer(new OpenCLMemory(TNN_CL_BUFFER));
    int buffer_size = DimsFunctionUtils::GetDim(dims, 0) * ROUND_UP(DimsFunctionUtils::GetDim(dims, 1), 4) *
                      DimsFunctionUtils::GetDim(dims, 2) * DimsFunctionUtils::GetDim(dims, 3);
    cl_int ret = CL_SUCCESS;
    cl::Buffer param_clbuffer(*opencl_runtime->Context(), CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                              buffer_size * sizeof(float), nullptr, &ret);
    if (ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(ret)
        return Status(TNNERR_OPENCL_MEMALLOC_ERROR, "OpenCL malloc memory failed");
    }
    buffer->SetData(&param_clbuffer);
    auto param_clbuffer_ptr = ocl_context_->CommandQueue()->enqueueMapBuffer(
        param_clbuffer, true, CL_MAP_WRITE, 0, buffer_size * sizeof(float), nullptr, nullptr, &ret);
    if (ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(ret)
        return Status(TNNERR_OPENCL_MEMMAP_ERROR, "OpenCL MemMap failed");
    }
    memset(param_clbuffer_ptr, 0, buffer_size * sizeof(float));
    ret = ocl_context_->CommandQueue()->enqueueUnmapMemObject(param_clbuffer, param_clbuffer_ptr);
    if (ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(ret)
        return Status(TNNERR_OPENCL_MEMUNMAP_ERROR, "OpenCL MemUnMap failed");
    }

    // create binary_param_
    int climage_w             = UP_DIV(DimsFunctionUtils::GetDim(dims, 1), 4) * DimsFunctionUtils::GetDim(dims, 3);
    int climage_h             = DimsFunctionUtils::GetDim(dims, 0) * DimsFunctionUtils::GetDim(dims, 2);
    cl_channel_type data_type = CL_FLOAT;
    if (opencl_runtime->GetPrecision() != PRECISION_HIGH)
        data_type = CL_HALF_FLOAT;
    cl::Image2D *image = new cl::Image2D(*opencl_runtime->Context(), CL_MEM_READ_WRITE,
                                         cl::ImageFormat(CL_RGBA, data_type), climage_w, climage_h, 0, nullptr, &ret);
    if (ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(ret)
        if (nullptr != image)
            delete image;
        return Status(TNNERR_OPENCL_MEMALLOC_ERROR, "OpenCL malloc memory failed");
    }
    inter_image_.reset(new OpenCLMemory(TNN_CL_IMAGE));
    inter_image_->SetData(image, true);

    // convert nchw buffer to Image
    ImageBufferConvertor convertor(opencl_runtime, ocl_context_->CommandQueue());
    return convertor.ConvertBufferToImage(buffer.get(), NCHW_BUFFER, dims, inter_image_.get(), true);
}

}  // namespace TNN_NS
