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

#include "tnn/device/opencl/acc/opencl_reshape_layer_acc.h"
#include "tnn/device/opencl/imagebuffer_convertor.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

Status OpenCLReshapeLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                   const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init Reshape Acc\n");
    Status ret = OpenCLLayerAcc::Init(context, param, resource, inputs, outputs);
    CHECK_TNN_OK(ret)

    int reshape_type = -1;
    ReshapeLayerParam *reshape_param = dynamic_cast<ReshapeLayerParam *>(param_);
    if (!reshape_param) {
        FlattenLayerParam *flatten_param = dynamic_cast<FlattenLayerParam *>(param_);
        if(!flatten_param) {
            LOGE("Error: layer param is null\n");
            return Status(TNNERR_MODEL_ERR, "Error: layer param is null");
        } else {
            reshape_type = 0;
        }
    } else {
        reshape_type = reshape_param->reshape_type;
    }

    run_3d_ndrange_ = false;
    op_name_        = "Reshape";

    auto input = inputs[0];
    auto output = outputs[0];

    auto input_dims = input->GetBlobDesc().dims;
    auto output_dims = output->GetBlobDesc().dims;
    input_dims_size_ = input_dims.size();
    output_dims_size_ = output_dims.size();

    std::string src_format = "Image", dst_format = "Image";
    im_to_bf_program_name_ = "image_to_buffer";
    bf_to_im_program_name_ = "buffer_to_image";
    src_format = input_dims_size_ == 5 ? "Image5D" : input_dims_size_ == 6 ? "Image6D" : src_format;
    im_to_bf_program_name_ = input_dims_size_ == 5 ? "image_5d_to_buffer" : input_dims_size_ == 6 ? "image_6d_to_buffer" : im_to_bf_program_name_;
    dst_format = output_dims_size_ == 5 ? "Image5D" : output_dims_size_ == 6 ? "Image6D" : dst_format;
    bf_to_im_program_name_ = output_dims_size_ == 5 ? "buffer_to_image_5d" : output_dims_size_ == 6 ? "buffer_to_image_6d" : bf_to_im_program_name_;

    if (reshape_type == 0)
    {
        im_to_bf_func_name_      = src_format + "ToNCHWBuffer";
        bf_to_im_func_name_      = "NCHWBufferTo" + dst_format;
    } else if (reshape_type == 1 && outputs[0]->GetBlobDesc().data_format == DATA_FORMAT_NHC4W4) {
        // tensorflow reshape data format is NHWC, only support NHC4W4 blob for now
        im_to_bf_func_name_      = src_format + "ToNHWCBuffer";
        bf_to_im_func_name_      = "NHWCBufferTo" + dst_format;
    } else {
        LOGE("Error: Unsupport reshape type(%d), src_format: %s, dst_format: %s\n",
             reshape_type, src_format.c_str(), dst_format.c_str());
        return Status(TNNERR_MODEL_ERR, "Error: OpenCLReshapeLayerAcc failed!\n");
    }

    execute_units_.resize(2);
    // image->buffer
    {
        std::set<std::string> build_opt;
        if (outputs[0]->GetBlobDesc().data_format == DATA_FORMAT_NCHW) {
            is_nchw_output_ = true;
            build_opt.emplace("-DENABLE_BUFFER_PRECISION_ADJUST");
        }
        build_opt.insert(build_options_.begin(), build_options_.end());
        ret = CreateExecuteUnit(execute_units_[0], im_to_bf_program_name_, im_to_bf_func_name_, build_opt);
        if (ret != TNN_OK) {
            LOGE("create execute unit failed!\n");
            return ret;
        }
    }

    // buffer->image
    {
        ret = CreateExecuteUnit(execute_units_[1], bf_to_im_program_name_, bf_to_im_func_name_, build_options_);
        if (ret != TNN_OK) {
            LOGE("create execute unit failed!\n");
            return ret;
        }
    }

    return TNN_OK;
}

OpenCLReshapeLayerAcc::~OpenCLReshapeLayerAcc() {}

Status OpenCLReshapeLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Reshape Acc Reshape\n");
    Status ret = OpenCLLayerAcc::Reshape(inputs, outputs);
    CHECK_TNN_OK(ret)
    auto input  = inputs[0];
    auto output = outputs[0];

    // reinit opencl execute unit if data format is changed during Reshape
    if (output->GetBlobDesc().data_format == DATA_FORMAT_NCHW && !is_nchw_output_) {
        std::set<std::string> build_opt;
        is_nchw_output_ = true;
        build_opt.emplace("-DENABLE_BUFFER_PRECISION_ADJUST");
        build_opt.insert(build_options_.begin(), build_options_.end());
        ret = CreateExecuteUnit(execute_units_[0], im_to_bf_program_name_, im_to_bf_func_name_, build_opt);
        CHECK_TNN_OK(ret)
    }

    auto input_dims  = input->GetBlobDesc().dims;
    auto output_dims = output->GetBlobDesc().dims;

    OpenCLRuntime *opencl_runtime = OpenCLRuntime::GetInstance();
    int blob_size                 = sizeof(float) * DimsVectorUtils::Count(input_dims);

    if (output->GetBlobDesc().data_format != DATA_FORMAT_NCHW) {
        inter_buffer_ = std::make_shared<cl::Buffer>(*opencl_runtime->Context(),
                                                     CL_MEM_READ_WRITE, blob_size);
    }

    // image->buffer
    {
        uint32_t idx = SetExecuteUnit2DSizeInfoDefault(execute_units_[0], input_dims);
        if (output->GetBlobDesc().data_format == DATA_FORMAT_NCHW) {
            execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Buffer *)output->GetHandle().base));
        } else {
            execute_units_[0].ocl_kernel.setArg(idx++, *inter_buffer_.get());
        }
        if (input_dims_size_ <= 4) {
            execute_units_[0].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(input_dims, 2)));
            execute_units_[0].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(input_dims, 3)));
            execute_units_[0].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(input_dims, 1)));
        } else if (input_dims_size_ == 5) {
            execute_units_[0].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(input_dims, 1)));
            execute_units_[0].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(input_dims, 2)));
            execute_units_[0].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(input_dims, 3)));
            execute_units_[0].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(input_dims, 4)));
        } else if (input_dims_size_ == 6) {
            execute_units_[0].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(input_dims, 1)));
            execute_units_[0].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(input_dims, 2)));
            execute_units_[0].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(input_dims, 3)));
            execute_units_[0].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(input_dims, 4)));
            execute_units_[0].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(input_dims, 5)));
        }
        execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)input->GetHandle().base));
    }

    // buffer->image
    if (output->GetBlobDesc().data_format == DATA_FORMAT_NCHW) {
        InsertUnactiveUnitId(1);
    } else {
        uint32_t idx = SetExecuteUnit2DSizeInfoDefault(execute_units_[1], output_dims);
        execute_units_[1].ocl_kernel.setArg(idx++, *inter_buffer_.get());
        if (output_dims_size_ <= 4) {
            execute_units_[1].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(output_dims, 2)));
            execute_units_[1].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(output_dims, 3)));
            execute_units_[1].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(output_dims, 1)));
        } else if (output_dims_size_ == 5) {
            execute_units_[1].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(output_dims, 1)));
            execute_units_[1].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(output_dims, 2)));
            execute_units_[1].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(output_dims, 3)));
            execute_units_[1].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(output_dims, 4)));
        } else if (output_dims_size_ == 6) {
            execute_units_[1].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(output_dims, 1)));
            execute_units_[1].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(output_dims, 2)));
            execute_units_[1].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(output_dims, 3)));
            execute_units_[1].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(output_dims, 4)));
            execute_units_[1].ocl_kernel.setArg(idx++, static_cast<uint32_t>(DimsFunctionUtils::GetDim(output_dims, 5)));
        }
        execute_units_[1].ocl_kernel.setArg(idx++, *((cl::Image *)output->GetHandle().base));
    }

    return TNN_OK;
}

std::vector<DataFormat> OpenCLReshapeLayerAcc::SupportDataFormat(DataType data_type,
                                                                 int dims_size,
                                                                 BlobType blob_type) {
    std::vector<DataFormat> support_list;
    if (data_type == DATA_TYPE_INT32) {
        // reshape layer blob may contain shape info
        support_list.push_back(DATA_FORMAT_NHC4W4);
    } else if (dims_size >= 2 && dims_size <= 6) { // only support up to 6 dims
        support_list.push_back(DATA_FORMAT_NHC4W4);
        // output blob support nchw
        if (blob_type == BLOB_OUTPUT) {
            support_list.push_back(DATA_FORMAT_NCHW);
        }
    }
    return support_list;
}

std::vector<DataType> OpenCLReshapeLayerAcc::SupportDataType(int dims_size, BlobType blob_type) {
    if (blob_type == BLOB_INPUT) {
        // reshape layer blob may contain shape info
        return {DATA_TYPE_FLOAT, DATA_TYPE_HALF, DATA_TYPE_INT32};
    } else {
        return {DATA_TYPE_FLOAT, DATA_TYPE_HALF};
    }
}

REGISTER_OPENCL_ACC(Reshape, LAYER_RESHAPE)
REGISTER_OPENCL_ACC(Reshape, LAYER_FLATTEN)
REGISTER_OPENCL_LAYOUT(LAYER_RESHAPE, DATA_FORMAT_NHC4W4);
REGISTER_OPENCL_LAYOUT(LAYER_FLATTEN, DATA_FORMAT_NHC4W4);

}  // namespace TNN_NS
