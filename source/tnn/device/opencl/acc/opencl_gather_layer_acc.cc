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

#include "tnn/device/opencl/acc/opencl_layer_acc.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

class OpenCLGatherLayerAcc : public OpenCLLayerAcc {
public:
    virtual Status Init(Context *context, LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                        const std::vector<Blob *> &outputs) override;

    virtual ~OpenCLGatherLayerAcc() override;

    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;

private:
    virtual std::vector<DataType> SupportDataType(int dims_size, BlobType blob_type) override;

    Status ConvertIndicesBuffer(RawBuffer& indices);

    Status ConvertDataBuffer(RawBuffer& data);

    std::shared_ptr<cl::Buffer> src_buffer_ = nullptr;
    std::shared_ptr<cl::Buffer> dst_buffer_ = nullptr;
    std::shared_ptr<cl::Buffer> indices_buffer_ = nullptr;
};

Status OpenCLGatherLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                               const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init Gather Acc\n");
    Status ret = OpenCLLayerAcc::Init(context, param, resource, inputs, outputs);
    CHECK_TNN_OK(ret)

    auto layer_param = dynamic_cast<GatherLayerParam*>(param);
    CHECK_PARAM_NULL(layer_param);
    int axis = layer_param->axis;

    run_3d_ndrange_ = true;
    op_name_        = "Gather";

    auto layer_resource = dynamic_cast<GatherLayerResource*>(resource_);

    if (!layer_param->data_in_resource && layer_param->indices_in_resource && layer_resource) {
        ret = ConvertIndicesBuffer(layer_resource->indices);
        auto input = inputs[0];
        auto output = outputs[0];
        auto input_dims = input->GetBlobDesc().dims;
        auto output_dims = output->GetBlobDesc().dims;
        auto input_dims_size = input_dims.size();
        auto output_dims_size = output_dims.size();

        std::string src_format = "Image", dst_format = "Image";
        std::string img_to_buf_program_name = "image_to_buffer", buf_to_img_program_name = "buffer_to_image";
        src_format = input_dims_size == 5 ? "Image5D" : input_dims_size == 6 ? "Image6D" : src_format;
        img_to_buf_program_name = input_dims_size == 5 ? "image_5d_to_buffer" : input_dims_size == 6 ? "image_6d_to_buffer" : img_to_buf_program_name;
        dst_format = output_dims_size == 5 ? "Image5D" : output_dims_size == 6 ? "Image6D" : dst_format;
        buf_to_img_program_name = output_dims_size == 5 ? "buffer_to_image_5d" : output_dims_size == 6 ? "buffer_to_image_6d" : buf_to_img_program_name;

        // create kernel
        execute_units_.resize(3);
        // image->buffer
        {
            ret = CreateExecuteUnit(execute_units_[0], img_to_buf_program_name, src_format + "ToNCHWBuffer",
                                    build_options_);
            if (ret != TNN_OK) {
                LOGE("create execute unit failed!\n");
                return ret;
            }
        }

        // gather
        {
            ret = CreateExecuteUnit(execute_units_[1], "gather", "GatherCommon", build_options_);
            if (ret != TNN_OK) {
                LOGE("create execute unit failed!\n");
                return ret;
            }
        }

        // buffer->image
        {
            ret = CreateExecuteUnit(execute_units_[2], buf_to_img_program_name, "NCHWBufferTo" + dst_format,
                                    build_options_);
            if (ret != TNN_OK) {
                LOGE("create execute unit failed!\n");
                return ret;
            }
        }
        return TNN_OK; 
    } else if(layer_param->data_in_resource && !layer_param->indices_in_resource && layer_resource) {
        DimsVector input_data_dims = layer_resource->data.GetBufferDims();
        
        ret = ConvertDataBuffer(layer_resource->data);
        auto input = inputs[0];
        auto output = outputs[0];
        auto input_dims = input->GetBlobDesc().dims;
        auto output_dims = output->GetBlobDesc().dims;
        auto input_dims_size = input_dims.size();
        auto output_dims_size = output_dims.size();

        std::string src_format = "Image", dst_format = "Image";
        std::string img_to_buf_program_name = "image_to_buffer", buf_to_img_program_name = "buffer_to_image";


        // create kernel
        execute_units_.resize(3);
        // image->buffer
        {
            ret = CreateExecuteUnit(execute_units_[0], img_to_buf_program_name, src_format + "ToNCHWBuffer",
                                    build_options_);
            if (ret != TNN_OK) {
                LOGE("create execute unit failed!\n");
                return ret;
            }
        }

        // gather
        {
            ret = CreateExecuteUnit(execute_units_[1], "gather", "GatherCommon", build_options_);
            if (ret != TNN_OK) {
                LOGE("create execute unit failed!\n");
                return ret;
            }
        }

        // buffer->image
        {
            ret = CreateExecuteUnit(execute_units_[2], buf_to_img_program_name, "NCHWBufferTo" + dst_format,
                                    build_options_);
            if (ret != TNN_OK) {
                LOGE("create execute unit failed!\n");
                return ret;
            }
        }
        return TNN_OK; 
    } else {
        return Status(TNNERR_PARAM_ERR, "Error: only support indices in resource now \n"); 
    }
}

Status OpenCLGatherLayerAcc::ConvertIndicesBuffer(RawBuffer& indices) {
    OpenCLRuntime *opencl_runtime = OpenCLRuntime::GetInstance();
    cl_int ret = CL_SUCCESS;
    indices_buffer_ = std::make_shared<cl::Buffer>(*opencl_runtime->Context(), CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                      DimsVectorUtils::Count(indices.GetBufferDims()) * sizeof(int), nullptr, &ret);
    if (ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(ret)
        return Status(TNNERR_OPENCL_MEMALLOC_ERROR, "OpenCL malloc memory failed");
    }
    auto indices_clbuffer_ptr = ocl_context_->CommandQueue()->enqueueMapBuffer(
        *indices_buffer_.get(), true, CL_MAP_WRITE, 0, DimsVectorUtils::Count(indices.GetBufferDims()) * sizeof(int), nullptr, nullptr, &ret);
    if (ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(ret)
        return Status(TNNERR_OPENCL_MEMMAP_ERROR, "OpenCL MemMap failed");
    }
    memcpy(indices_clbuffer_ptr, indices.force_to<char*>(), DimsVectorUtils::Count(indices.GetBufferDims()) * sizeof(int));
    ret = ocl_context_->CommandQueue()->enqueueUnmapMemObject(*indices_buffer_.get(), indices_clbuffer_ptr);
    if (ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(ret)
        return Status(TNNERR_OPENCL_MEMUNMAP_ERROR, "OpenCL MemUnMap failed");
    }
    return TNN_OK;
}

Status OpenCLGatherLayerAcc::ConvertDataBuffer(RawBuffer& data) {
    OpenCLRuntime *opencl_runtime = OpenCLRuntime::GetInstance();
    cl_int ret = CL_SUCCESS;
    indices_buffer_ = std::make_shared<cl::Buffer>(*opencl_runtime->Context(), CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                      DimsVectorUtils::Count(data.GetBufferDims()) * sizeof(float), nullptr, &ret);
    if (ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(ret)
        return Status(TNNERR_OPENCL_MEMALLOC_ERROR, "OpenCL malloc memory failed");
    }
    auto indices_clbuffer_ptr = ocl_context_->CommandQueue()->enqueueMapBuffer(
        *indices_buffer_.get(), true, CL_MAP_WRITE, 0, DimsVectorUtils::Count(data.GetBufferDims()) * sizeof(float), nullptr, nullptr, &ret);
    if (ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(ret)
        return Status(TNNERR_OPENCL_MEMMAP_ERROR, "OpenCL MemMap failed");
    }
    memcpy(indices_clbuffer_ptr, data.force_to<char*>(), DimsVectorUtils::Count(data.GetBufferDims()) * sizeof(float));
    ret = ocl_context_->CommandQueue()->enqueueUnmapMemObject(*indices_buffer_.get(), indices_clbuffer_ptr);
    if (ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(ret)
        return Status(TNNERR_OPENCL_MEMUNMAP_ERROR, "OpenCL MemUnMap failed");
    }
    return TNN_OK;
}

OpenCLGatherLayerAcc::~OpenCLGatherLayerAcc() {}

Status OpenCLGatherLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Gather Acc Reshape\n");
    auto layer_param = dynamic_cast<GatherLayerParam*>(param_);
    Status ret = OpenCLLayerAcc::Reshape(inputs, outputs);
    CHECK_TNN_OK(ret)

    auto layer_resource = dynamic_cast<GatherLayerResource*>(resource_);

    if(!layer_param->data_in_resource && layer_param->indices_in_resource) {
    auto input  = inputs[0];
    auto output = outputs[0];

    auto input_dims  = input->GetBlobDesc().dims;
    auto output_dims = output->GetBlobDesc().dims;

    OpenCLRuntime *opencl_runtime = OpenCLRuntime::GetInstance();
    int input_size                 = sizeof(float) * DimsVectorUtils::Count(input_dims);
    int output_size                = sizeof(float) * DimsVectorUtils::Count(output_dims);

    src_buffer_ = std::make_shared<cl::Buffer>(*opencl_runtime->Context(),
                                                     CL_MEM_READ_WRITE, input_size);
    dst_buffer_ = std::make_shared<cl::Buffer>(*opencl_runtime->Context(),
                                                     CL_MEM_READ_WRITE, output_size);

    // image->buffer
    {
        uint32_t idx = SetExecuteUnit2DSizeInfoDefault(execute_units_[0], input_dims);
        execute_units_[0].ocl_kernel.setArg(idx++, *src_buffer_.get());
        if (input_dims.size() <= 4) {
            execute_units_[0].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(input_dims, 2)));
            execute_units_[0].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(input_dims, 3)));
            execute_units_[0].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(input_dims, 1)));
        } else if (input_dims.size() == 5) {
            execute_units_[0].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(input_dims, 1)));
            execute_units_[0].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(input_dims, 2)));
            execute_units_[0].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(input_dims, 3)));
            execute_units_[0].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(input_dims, 4)));
        } else if (input_dims.size()== 6) {
            execute_units_[0].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(input_dims, 1)));
            execute_units_[0].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(input_dims, 2)));
            execute_units_[0].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(input_dims, 3)));
            execute_units_[0].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(input_dims, 4)));
            execute_units_[0].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(input_dims, 5)));
        }
        
        execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)input->GetHandle().base));
    }

    //gather
    {
        int axis = layer_param->axis;
        int inner_size = DimsVectorUtils::Count(input_dims, axis + 1);
        int outer_size = DimsVectorUtils::Count(input_dims, 0, axis);
        int input_axis_size = input_dims[axis];
        int indice_size = DimsVectorUtils::Count(output_dims) / inner_size / outer_size;
        int output_outer_step = inner_size * indice_size;
        int input_outer_step = inner_size * input_axis_size;
        execute_units_[1].global_work_size = {static_cast<uint32_t>(inner_size), static_cast<uint32_t>(indice_size), static_cast<uint32_t>(outer_size)};
        execute_units_[1].local_work_size = LocalWS3DDefault(execute_units_[1]); 
        uint32_t idx = 0;
        execute_units_[1].ocl_kernel.setArg(idx++, execute_units_[1].global_work_size[0]);
        execute_units_[1].ocl_kernel.setArg(idx++, execute_units_[1].global_work_size[1]);
        execute_units_[1].ocl_kernel.setArg(idx++, execute_units_[1].global_work_size[2]);
        if(layer_param->data_in_resource && !layer_param->indices_in_resource) {
            execute_units_[1].ocl_kernel.setArg(idx++, *indices_buffer_);
            execute_units_[1].ocl_kernel.setArg(idx++, *src_buffer_);
        } else if(!layer_param->data_in_resource && layer_param->indices_in_resource){
            execute_units_[1].ocl_kernel.setArg(idx++, *src_buffer_);
            execute_units_[1].ocl_kernel.setArg(idx++, *indices_buffer_);
        }
        execute_units_[1].ocl_kernel.setArg(idx++, *dst_buffer_);
        execute_units_[1].ocl_kernel.setArg(idx++, inner_size);
        execute_units_[1].ocl_kernel.setArg(idx++, input_outer_step);
        execute_units_[1].ocl_kernel.setArg(idx++, output_outer_step);
    }

    // buffer->image
    {
        uint32_t idx = SetExecuteUnit2DSizeInfoDefault(execute_units_[2], output_dims);
        execute_units_[2].ocl_kernel.setArg(idx++, *dst_buffer_.get());
        if (output_dims.size() <= 4) {
            execute_units_[2].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(output_dims, 2)));
            execute_units_[2].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(output_dims, 3)));
            execute_units_[2].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(output_dims, 1)));
        } else if (output_dims.size() == 5) {
            execute_units_[2].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(output_dims, 1)));
            execute_units_[2].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(output_dims, 2)));
            execute_units_[2].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(output_dims, 3)));
            execute_units_[2].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(output_dims, 4)));
        } else if (output_dims.size() == 6) {
            execute_units_[2].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(output_dims, 1)));
            execute_units_[2].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(output_dims, 2)));
            execute_units_[2].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(output_dims, 3)));
            execute_units_[2].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(output_dims, 4)));
            execute_units_[2].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(output_dims, 5)));
        }
        execute_units_[2].ocl_kernel.setArg(idx++, *((cl::Image *)output->GetHandle().base));
    }
    } else if(layer_param->data_in_resource && !layer_param->indices_in_resource) {
        auto input  = inputs[0];
        auto output = outputs[0];

        auto input_dims  = input->GetBlobDesc().dims;
        auto input_data_dims = layer_resource->data.GetBufferDims();
        auto output_dims = output->GetBlobDesc().dims;

        OpenCLRuntime *opencl_runtime = OpenCLRuntime::GetInstance();
        int input_size                 = sizeof(float) * DimsVectorUtils::Count(input_dims);
        int input_data_size            = sizeof(float) * DimsVectorUtils::Count(input_data_dims);
        int output_size                = sizeof(float) * DimsVectorUtils::Count(output_dims);

        src_buffer_ = std::make_shared<cl::Buffer>(*opencl_runtime->Context(),
                                                     CL_MEM_READ_WRITE, input_data_size);
        dst_buffer_ = std::make_shared<cl::Buffer>(*opencl_runtime->Context(),
                                                     CL_MEM_READ_WRITE, output_size);

        // image->buffer, for indices
        {
            uint32_t idx = SetExecuteUnit2DSizeInfoDefault(execute_units_[0], input_dims);
            execute_units_[0].ocl_kernel.setArg(idx++, *src_buffer_.get());
            if (input_dims.size() <= 4) {
                execute_units_[0].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(input_dims, 2)));
                execute_units_[0].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(input_dims, 3)));
                execute_units_[0].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(input_dims, 1)));
            } else if (input_dims.size() == 5) {
                execute_units_[0].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(input_dims, 1)));
                execute_units_[0].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(input_dims, 2)));
                execute_units_[0].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(input_dims, 3)));
                execute_units_[0].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(input_dims, 4)));
            } else if (input_dims.size()== 6) {
                execute_units_[0].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(input_dims, 1)));
                execute_units_[0].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(input_dims, 2)));
                execute_units_[0].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(input_dims, 3)));
                execute_units_[0].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(input_dims, 4)));
                execute_units_[0].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(input_dims, 5)));
            }
            execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)input->GetHandle().base));
        }

        //gather
        {
            int axis = layer_param->axis;
            int inner_size = DimsVectorUtils::Count(input_data_dims, axis + 1);
            int outer_size = DimsVectorUtils::Count(input_data_dims, 0, axis);
            int input_axis_size = input_data_dims[axis];
            int indice_size = DimsVectorUtils::Count(output_dims) / inner_size / outer_size;
            int output_outer_step = inner_size * indice_size;
            int input_outer_step = inner_size * input_axis_size;
            execute_units_[1].global_work_size = {static_cast<uint32_t>(inner_size), static_cast<uint32_t>(indice_size), static_cast<uint32_t>(outer_size)};
            execute_units_[1].local_work_size = LocalWS3DDefault(execute_units_[1]); 
            uint32_t idx = 0;
            execute_units_[1].ocl_kernel.setArg(idx++, execute_units_[1].global_work_size[0]);
            execute_units_[1].ocl_kernel.setArg(idx++, execute_units_[1].global_work_size[1]);
            execute_units_[1].ocl_kernel.setArg(idx++, execute_units_[1].global_work_size[2]);
            
            execute_units_[1].ocl_kernel.setArg(idx++, *indices_buffer_);
            execute_units_[1].ocl_kernel.setArg(idx++, *src_buffer_);
            
            execute_units_[1].ocl_kernel.setArg(idx++, *dst_buffer_);
            execute_units_[1].ocl_kernel.setArg(idx++, inner_size);
            execute_units_[1].ocl_kernel.setArg(idx++, input_outer_step);
            execute_units_[1].ocl_kernel.setArg(idx++, output_outer_step);
        }

        // buffer->image
        {
            uint32_t idx = SetExecuteUnit2DSizeInfoDefault(execute_units_[2], output_dims);
            execute_units_[2].ocl_kernel.setArg(idx++, *dst_buffer_.get());
            if (output_dims.size() <= 4) {
                execute_units_[2].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(output_dims, 2)));
                execute_units_[2].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(output_dims, 3)));
                execute_units_[2].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(output_dims, 1)));
            } else if (output_dims.size() == 5) {
                execute_units_[2].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(output_dims, 1)));
                execute_units_[2].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(output_dims, 2)));
                execute_units_[2].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(output_dims, 3)));
                execute_units_[2].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(output_dims, 4)));
            } else if (output_dims.size() == 6) {
                execute_units_[2].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(output_dims, 1)));
                execute_units_[2].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(output_dims, 2)));
                execute_units_[2].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(output_dims, 3)));
                execute_units_[2].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(output_dims, 4)));
                execute_units_[2].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(output_dims, 5)));
            }
            execute_units_[2].ocl_kernel.setArg(idx++, *((cl::Image *)output->GetHandle().base));
        }
    } else {
        LOGD("gather failed\n");
    }
    return TNN_OK;
}

std::vector<DataType> OpenCLGatherLayerAcc::SupportDataType(int dims_size, BlobType blob_type) {
        return {DATA_TYPE_FLOAT, DATA_TYPE_HALF, DATA_TYPE_INT32};
}

REGISTER_OPENCL_ACC(Gather, LAYER_GATHER)
REGISTER_OPENCL_LAYOUT(LAYER_GATHER, DATA_FORMAT_NHC4W4);

}  // namespace TNN_NS
