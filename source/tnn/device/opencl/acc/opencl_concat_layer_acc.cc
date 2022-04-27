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
#include "tnn/device/opencl/imagebuffer_convertor.h"
#include "tnn/device/opencl/acc/opencl_reshape_layer_acc.h"

#include "tnn/utils/dims_utils.h"
#include "tnn/utils/string_utils_inner.h"

namespace TNN_NS {

typedef enum { BUFFER_COPY = 0, IMAGE_COPY, TWO_INPUTS_CHANNEL_4X, TWO_INPUTS_CHANNEL_MOD_123 } ConcatKernelType;

class OpenCLConcatLayerAcc : public OpenCLLayerAcc {
public:
    virtual Status Init(Context *context, LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                        const std::vector<Blob *> &outputs) override;

    virtual ~OpenCLConcatLayerAcc() override;

    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;

    virtual Status Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;
#if TNN_PROFILE
    virtual double GetBandwidth() override;
#endif

private:
    Status ReshapeImageConcat(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
    Status ReshapeTwoInputsConcat(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
    Status ReshapeBufferConcat(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

    // add reshape when the dims > 4
    Status InitReshapeLayer(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
    // calculate the shape in reshape
    DimsVector CalculateShape(const DimsVector &dims);
    int CalculateAxis(int axis, int dims_size);

private:
    std::shared_ptr<cl::Buffer> output_buffer_              = nullptr;
    std::vector<std::shared_ptr<cl::Buffer>> input_buffers_ = {};
    int axis_                                               = 1;
    bool do_image_concat_                                   = true;
    ConcatKernelType concat_type_                           = BUFFER_COPY;

    // add reshape when the dims > 4
    bool need_reshape_                                                          = false;
    shared_ptr<OpenCLReshapeLayerAcc> output_reshape_layer_acc_                 = nullptr;
    std::vector<shared_ptr<OpenCLReshapeLayerAcc>> input_reshape_layer_acc_vec_ = {};
    std::vector<Blob *> concat_inputs_                                          = {};
    std::vector<Blob *> concat_outputs_                                         = {};
    std::vector<shared_ptr<Blob>> concat_input_blob_vec_                        = {};
    std::vector<shared_ptr<cl::Image2D>> concat_input_image_vec_                = {};
    std::vector<std::shared_ptr<ReshapeLayerParam>> reshape_param_vec_          = {};
    std::vector<std::vector<Blob *>> input_reshape_inputs_                      = {};
    std::vector<std::vector<Blob *>> input_reshape_outputs_                     = {};
    shared_ptr<Blob> concat_output_blob_                                        = nullptr;
    shared_ptr<cl::Image2D> concat_output_image_                                = nullptr;
};

bool CheckIsTwoInputs(const size_t input_size, const int axis) {
    return input_size == 2 && axis == 1;
}

Status OpenCLConcatLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                  const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init Concat Acc\n");
    Status ret = OpenCLLayerAcc::Init(context, param, resource, inputs, outputs);
    CHECK_TNN_OK(ret)

    run_3d_ndrange_ = false;
    op_name_        = "Concat";

    ConcatLayerParam *concat_param = dynamic_cast<ConcatLayerParam *>(param);
    CHECK_PARAM_NULL(concat_param);

    axis_            = CalculateAxis(concat_param->axis, outputs[0]->GetBlobDesc().dims.size());
    need_reshape_    = outputs[0]->GetBlobDesc().dims.size() > 4;
    do_image_concat_ = true;
    if (axis_ == 1) {
        for (size_t i = 0; i < inputs.size() - 1; ++i) {
            int channel = DimsFunctionUtils::GetDim(inputs[i]->GetBlobDesc().dims, 1);
            if (channel % 4 != 0) {
                do_image_concat_ = false;
                break;
            }
        }
    }

    LOGD("do_image_concat: %s\n", do_image_concat_ ? "true" : "false");

    // choose kernel type
    if (CheckIsTwoInputs(inputs.size(), axis_)) {
        if (do_image_concat_) {
            if (gpu_info_.type == ADRENO) {
                concat_type_ = TWO_INPUTS_CHANNEL_4X;
            } else {
                concat_type_ = IMAGE_COPY;
            }
        } else {
            concat_type_ = TWO_INPUTS_CHANNEL_MOD_123;
        }
    } else {
        if (do_image_concat_) {
            concat_type_ = IMAGE_COPY;
        } else {
            concat_type_ = BUFFER_COPY;
        }
    }

    if (need_reshape_) {
        concat_type_     = BUFFER_COPY;
    }

    // create kernel
    std::string kernel_name;

    if (IMAGE_COPY == concat_type_) {
        std::string program_name = "copy";
        execute_units_.resize(inputs.size());
        for (size_t i = 0; i < execute_units_.size(); i++) {
            kernel_name = "CopyImage";
            ret         = CreateExecuteUnit(execute_units_[i], program_name, kernel_name, build_options_);
            if (ret != TNN_OK) {
                return ret;
            }
        }
    } else if (TWO_INPUTS_CHANNEL_4X == concat_type_) {
        std::string program_name = "concat";
        kernel_name              = "ConcatChannel4X";
        execute_units_.resize(1);
        ret = CreateExecuteUnit(execute_units_[0], program_name, kernel_name, build_options_);
        if (ret != TNN_OK) {
            return ret;
        }
    } else if (TWO_INPUTS_CHANNEL_MOD_123 == concat_type_) {
        std::set<std::string> build_options;
        build_options.emplace("-DCHANNEL0_MOD_4=" +
            ToString(DimsFunctionUtils::GetDim(inputs[0]->GetBlobDesc().dims, 1) % 4));
        std::string program_name = "concat";
        kernel_name              = "ConcatChannel";
        execute_units_.resize(1);
        build_options.insert(build_options_.begin(), build_options_.end());
        ret = CreateExecuteUnit(execute_units_[0], program_name, kernel_name, build_options);
        if (ret != TNN_OK) {
            return ret;
        }
    } else {
        // default BUFFER_COPY
        std::string program_name = "copy";
        execute_units_.resize(2 * inputs.size() + 1);
        for (size_t i = 0; i < inputs.size(); i++) {
            // Image to Buffer
            kernel_name = "CopyImageToBuffer";
            ret         = CreateExecuteUnit(execute_units_[2 * i], program_name, kernel_name, build_options_);
            if (ret != TNN_OK) {
                return ret;
            }
            // Merge Buffer to Buffer
            kernel_name = "CopyBuffer";
            ret         = CreateExecuteUnit(execute_units_[2 * i + 1], program_name, kernel_name, build_options_);
            if (ret != TNN_OK) {
                return ret;
            }
        }
        // Buffer to Image
        kernel_name = "CopyBufferToImage";
        ret         = CreateExecuteUnit(execute_units_[2 * inputs.size()], program_name, kernel_name, build_options_);
        if (ret != TNN_OK) {
            return ret;
        }
    }

    return TNN_OK;
}

OpenCLConcatLayerAcc::~OpenCLConcatLayerAcc() {}

Status OpenCLConcatLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Concat Acc Reshape\n");
    Status ret = OpenCLLayerAcc::Reshape(inputs, outputs);
    CHECK_TNN_OK(ret)

    if (need_reshape_) {
        ret = InitReshapeLayer(inputs, outputs);
        CHECK_TNN_OK(ret)
    }

    auto concat_inputs  = need_reshape_ ? concat_inputs_ : inputs;
    auto concat_outputs = need_reshape_ ? concat_outputs_ : outputs;

    if (need_reshape_) {
        const int input_size = input_reshape_layer_acc_vec_.size();
        for (int i = 0; i < input_size; i++) {
            if (input_reshape_layer_acc_vec_[i] == nullptr) {
                return Status(TNNERR_OPENCL_ACC_RESHAPE_ERROR, "reshape layer acc in Concat is null");
            }
            ret = input_reshape_layer_acc_vec_[i]->Reshape(input_reshape_inputs_[i], input_reshape_outputs_[i]);
            CHECK_TNN_OK(ret)
        }
    }

    if (IMAGE_COPY == concat_type_) {
        ret = ReshapeImageConcat(concat_inputs, concat_outputs);
    } else if (TWO_INPUTS_CHANNEL_4X == concat_type_) {
        ret = ReshapeTwoInputsConcat(concat_inputs, concat_outputs);
    } else if (TWO_INPUTS_CHANNEL_MOD_123 == concat_type_) {
        ret = ReshapeTwoInputsConcat(concat_inputs, concat_outputs);
    } else {
        // default BUFFER_COPY
        ret = ReshapeBufferConcat(concat_inputs, concat_outputs);
    }

    if (need_reshape_) {
        if (output_reshape_layer_acc_ == nullptr) {
            return Status(TNNERR_OPENCL_ACC_RESHAPE_ERROR, "reshape layer acc in Concat is null");
        }
        ret = output_reshape_layer_acc_->Reshape(concat_outputs_, outputs);
        CHECK_TNN_OK(ret)
    }

    return ret;
}

Status OpenCLConcatLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    Status ret          = TNN_OK;
    auto concat_inputs  = need_reshape_ ? concat_inputs_ : inputs;
    auto concat_outputs = need_reshape_ ? concat_outputs_ : outputs;

    if (need_reshape_) {
        const int input_size = input_reshape_layer_acc_vec_.size();
        for (int i = 0; i < input_size; i++) {
            if (input_reshape_layer_acc_vec_[i] == nullptr) {
                return Status(TNNERR_OPENCL_ACC_RESHAPE_ERROR, "reshape layer acc in Concat is null");
            }
            ret = input_reshape_layer_acc_vec_[i]->Forward(input_reshape_inputs_[i], input_reshape_outputs_[i]);
            CHECK_TNN_OK(ret)
        }
    }

    ret = OpenCLLayerAcc::Forward(concat_inputs, concat_outputs);

    if (need_reshape_) {
        if (output_reshape_layer_acc_ == nullptr) {
            return Status(TNNERR_OPENCL_ACC_RESHAPE_ERROR, "reshape layer acc in Concat is null");
        }
        ret = output_reshape_layer_acc_->Forward(concat_outputs, outputs);
        CHECK_TNN_OK(ret)
    }

    return ret;
}

#if TNN_PROFILE
double OpenCLConcatLayerAcc::GetBandwidth() {
    OpenCLRuntime *opencl_runtime = OpenCLRuntime::GetInstance();
    int data_type_size            = opencl_runtime->GetPrecision() != PRECISION_HIGH ? 2 : 4;
    return DimsVectorUtils::Count(output_dims_) * data_type_size / 1000.0 / 1000.0;
}
#endif

Status OpenCLConcatLayerAcc::ReshapeImageConcat(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto output      = outputs[0];
    auto output_dims = output->GetBlobDesc().dims;

    // output_width, output_height
    int output_wh[]     = {DimsFunctionUtils::GetDim(output_dims, 3), DimsFunctionUtils::GetDim(output_dims, 2)};
    int input_offset[]  = {0, 0, 0, 0};
    int output_offset[] = {0, 0, 0, 0};

    for (int i = 0; i < execute_units_.size(); ++i) {
        auto input      = inputs[i];
        auto input_dims = input->GetBlobDesc().dims;
        // input_width, input_height
        int input_wh[] = {DimsFunctionUtils::GetDim(input_dims, 3), DimsFunctionUtils::GetDim(input_dims, 2)};
        // batch, input_channel/4, input_height, input_width
        int region[] = {DimsFunctionUtils::GetDim(input_dims, 0),
                        UP_DIV(DimsFunctionUtils::GetDim(input_dims, 1), 4),
                        DimsFunctionUtils::GetDim(input_dims, 2), DimsFunctionUtils::GetDim(input_dims, 3)};

        auto &unit = execute_units_[i];
        int idx    = SetExecuteUnit2DSizeInfoDefault(unit, input_dims);
        unit.ocl_kernel.setArg(idx++, *((cl::Image *)input->GetHandle().base));
        unit.ocl_kernel.setArg(idx++, *((cl::Image *)output->GetHandle().base));
        unit.ocl_kernel.setArg(idx++, input_offset);
        unit.ocl_kernel.setArg(idx++, output_offset);
        unit.ocl_kernel.setArg(idx++, input_wh);
        unit.ocl_kernel.setArg(idx++, output_wh);
        unit.ocl_kernel.setArg(idx++, input_wh);

        output_offset[axis_] += region[axis_];
    }

    return TNN_OK;
}

Status OpenCLConcatLayerAcc::ReshapeBufferConcat(const std::vector<Blob *> &inputs,
                                                 const std::vector<Blob *> &outputs) {
    OpenCLRuntime *opencl_runtime = OpenCLRuntime::GetInstance();
    auto output                   = outputs[0];
    auto output_dims              = output->GetBlobDesc().dims;

    // allocate temp buffers
    int blob_elem_size = opencl_runtime->GetPrecision() != PRECISION_HIGH ? 2 : 4;
    int output_size    = DimsVectorUtils::Count(output->GetBlobDesc().dims);
    output_buffer_ =
        std::make_shared<cl::Buffer>(*opencl_runtime->Context(), CL_MEM_READ_WRITE, output_size * blob_elem_size);
    input_buffers_.resize(inputs.size());
    for (size_t i = 0; i < inputs.size(); i++) {
        int input_size = DimsVectorUtils::Count(inputs[i]->GetBlobDesc().dims);
        input_buffers_[i] =
            std::make_shared<cl::Buffer>(*opencl_runtime->Context(), CL_MEM_READ_WRITE, input_size * blob_elem_size);
    }

    // set args
    int input_offset[]  = {0, 0, 0, 0};
    int output_offset[] = {0, 0, 0, 0};
    int output_stride[] = {DimsFunctionUtils::GetDim(output_dims, 1) *
                           DimsFunctionUtils::GetDim(output_dims, 3) * DimsFunctionUtils::GetDim(output_dims, 2), 1,
                           DimsFunctionUtils::GetDim(output_dims, 3) * DimsFunctionUtils::GetDim(output_dims, 1),
                           DimsFunctionUtils::GetDim(output_dims, 1)};
    for (size_t i = 0; i < inputs.size(); i++) {
        auto input      = inputs[i];
        auto input_dims = input->GetBlobDesc().dims;

        const int batch        = DimsFunctionUtils::GetDim(input_dims, 0);
        const int input_height = DimsFunctionUtils::GetDim(input_dims, 2);
        const int input_width  = DimsFunctionUtils::GetDim(input_dims, 3);
        const int channels     = DimsFunctionUtils::GetDim(input_dims, 1);

        int input_wh[]      = {input_width, input_height};
        int buffer_region[] = {batch, channels, input_height, input_width};
        int input_stride[]  = {channels * input_width * input_height, 1, input_width * channels, channels};

        std::vector<int> buffer_output_size = {batch, channels, input_height, input_width};
        // image to buffer (from (NH,C4W4) to NHWC)
        {
            auto &unit = execute_units_[2 * i];
            int idx    = SetExecuteUnit2DSizeInfoDefault(unit, input_dims);
            unit.ocl_kernel.setArg(idx++, *((cl::Image *)input->GetHandle().base));
            unit.ocl_kernel.setArg(idx++, *(input_buffers_[i]));
            unit.ocl_kernel.setArg(idx++, input_offset);
            unit.ocl_kernel.setArg(idx++, input_offset);
            unit.ocl_kernel.setArg(idx++, input_wh);
            unit.ocl_kernel.setArg(idx++, input_stride);
            unit.ocl_kernel.setArg(idx++, 4 * sizeof(int), buffer_output_size.data());
        }

        // buffer to buffer
        {
            auto &unit = execute_units_[2 * i + 1];
            // special process: buffer_region[1] * 4 to get the right global size
            int idx = SetExecuteUnit2DSizeInfoDefault(
                unit, {buffer_region[0], buffer_region[1] * 4, buffer_region[2], buffer_region[3]});
            unit.ocl_kernel.setArg(idx++, *(input_buffers_[i]));
            unit.ocl_kernel.setArg(idx++, *output_buffer_);
            unit.ocl_kernel.setArg(idx++, input_offset);
            unit.ocl_kernel.setArg(idx++, output_offset);
            unit.ocl_kernel.setArg(idx++, input_stride);
            unit.ocl_kernel.setArg(idx++, output_stride);
            unit.ocl_kernel.setArg(idx++, input_wh);
        }
        output_offset[axis_] += buffer_region[axis_];
    }

    // buffer to image (from NHWC to (NH,C4W4))
    {
        int output_wh[] = {DimsFunctionUtils::GetDim(output_dims, 3), DimsFunctionUtils::GetDim(output_dims, 2)};
        auto &unit      = execute_units_[2 * inputs.size()];
        int idx         = SetExecuteUnit2DSizeInfoDefault(unit, output_dims);
        unit.ocl_kernel.setArg(idx++, *output_buffer_);
        unit.ocl_kernel.setArg(idx++, *((cl::Image *)output->GetHandle().base));
        unit.ocl_kernel.setArg(idx++, input_offset);
        unit.ocl_kernel.setArg(idx++, input_offset);
        unit.ocl_kernel.setArg(idx++, output_stride);
        unit.ocl_kernel.setArg(idx++, output_wh);
        unit.ocl_kernel.setArg(idx++, output_wh);
        unit.ocl_kernel.setArg(idx++, output_size - 1);
    }

    return TNN_OK;
}

Status OpenCLConcatLayerAcc::ReshapeTwoInputsConcat(const std::vector<Blob *> &inputs,
                                                    const std::vector<Blob *> &outputs) {
    run_3d_ndrange_  = true;
    auto output      = outputs[0];
    auto output_dims = output->GetBlobDesc().dims;
    auto input0      = inputs[0];
    auto input1      = inputs[1];

    // [output_channle/4, output_width, batch * output_height]
    execute_units_[0].global_work_size = {static_cast<uint32_t>(UP_DIV(DimsFunctionUtils::GetDim(output_dims, 1), 4)),
                                          static_cast<uint32_t>(DimsFunctionUtils::GetDim(output_dims, 3)),
                                          static_cast<uint32_t>(DimsFunctionUtils::GetDim(output_dims, 0) *
                                                                DimsFunctionUtils::GetDim(output_dims, 2))};
    execute_units_[0].local_work_size  = LocalWS3DDefault(execute_units_[0]);
    int idx                            = 0;
    execute_units_[0].ocl_kernel.setArg(idx++, execute_units_[0].global_work_size[0]);
    execute_units_[0].ocl_kernel.setArg(idx++, execute_units_[0].global_work_size[1]);
    execute_units_[0].ocl_kernel.setArg(idx++, execute_units_[0].global_work_size[2]);
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)input0->GetHandle().base));
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)input1->GetHandle().base));
    // input channel
    execute_units_[0].ocl_kernel.setArg(idx++, DimsFunctionUtils::GetDim(input0->GetBlobDesc().dims, 1));
    // output channel
    execute_units_[0].ocl_kernel.setArg(idx++, DimsFunctionUtils::GetDim(output_dims, 1));
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)output->GetHandle().base));
    return TNN_OK;
}

Status OpenCLConcatLayerAcc::InitReshapeLayer(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    Status ret = TNN_OK;

    // create input reshape
    concat_inputs_.clear();
    input_reshape_inputs_.clear();
    input_reshape_outputs_.clear();
    const int inputs_size = inputs.size();
    for (int i = 0; i < inputs_size; i++) {
        auto reshape_layer_acc = std::make_shared<OpenCLReshapeLayerAcc>();
        if (reshape_layer_acc == nullptr) {
            LOGE("Create Reshape Layer Acc in Concat failed!\n");
            return Status(TNNERR_CREATE_LAYER, "Create Reshape Layer Acc in Concat failed!");
        }
        input_reshape_layer_acc_vec_.push_back(reshape_layer_acc);

        BlobDesc desc            = inputs[i]->GetBlobDesc();
        desc.data_format         = DATA_FORMAT_NHC4W4;
        auto dims                = inputs[i]->GetBlobDesc().dims;
        auto shape               = CalculateShape(dims);
        desc.dims                = shape;
        auto reshape_output_blob = std::make_shared<Blob>(desc);
        if (reshape_output_blob == nullptr) {
            LOGE("Create reshape output blob in Concat failed!\n");
            return Status(TNNERR_CREATE_LAYER, "Create reshape output blob in Concat failed!");
        }
        concat_input_blob_vec_.push_back(reshape_output_blob);
        concat_inputs_.push_back(concat_input_blob_vec_[i].get());

        OpenCLRuntime *opencl_runtime = OpenCLRuntime::GetInstance();
        int climage_w = UP_DIV(DimsFunctionUtils::GetDim(desc.dims, 1), 4) * DimsFunctionUtils::GetDim(desc.dims, 3);
        int climage_h = DimsFunctionUtils::GetDim(desc.dims, 0) * DimsFunctionUtils::GetDim(desc.dims, 2);
        DimsVector imageshape{climage_w, climage_h};
        cl_channel_type data_type = CL_FLOAT;
        if (opencl_runtime->GetPrecision() != PRECISION_HIGH)
            data_type = CL_HALF_FLOAT;
        cl_int err                = CL_SUCCESS;
        auto reshape_output_image = std::make_shared<cl::Image2D>(*opencl_runtime->Context(), CL_MEM_READ_WRITE,
                                                                  cl::ImageFormat(CL_RGBA, data_type), imageshape[0],
                                                                  imageshape[1], 0, nullptr, &err);
        if (err != CL_SUCCESS) {
            CHECK_CL_SUCCESS(err)
            return Status(TNNERR_OPENCL_MEMALLOC_ERROR, "OpenCL malloc memory failed");
        }
        concat_input_image_vec_.push_back(reshape_output_image);

        BlobHandle blob_handle;
        blob_handle.base = concat_input_image_vec_[i].get();
        concat_input_blob_vec_[i]->SetHandle(blob_handle);

        // Init LayerAcc
        std::shared_ptr<ReshapeLayerParam> reshape_param = std::make_shared<ReshapeLayerParam>();
        reshape_param->type         = "Reshape";
        reshape_param->name         = layer_name_ + "_Input_Reshape_" + std::to_string(i);
        reshape_param->reshape_type = 0;
        reshape_param->axis         = 0;
        reshape_param->num_axes     = shape.size();
        reshape_param->shape        = shape;

        input_reshape_inputs_.push_back({inputs[i]});
        input_reshape_outputs_.push_back({concat_inputs_[i]});
        reshape_layer_acc->Init(ocl_context_, reshape_param.get(), nullptr, input_reshape_inputs_[i],
                                input_reshape_outputs_[i]);
        reshape_param_vec_.emplace_back(reshape_param);
    }

    // create output reshape
    {
        output_reshape_layer_acc_ = std::make_shared<OpenCLReshapeLayerAcc>();
        if (output_reshape_layer_acc_ == nullptr) {
            LOGE("Create Reshape Layer Acc in Concat failed!\n");
            return Status(TNNERR_CREATE_LAYER, "Create Reshape Layer Acc in Concat failed!");
        }

        BlobDesc desc                = outputs[0]->GetBlobDesc();
        desc.data_format             = DATA_FORMAT_NHC4W4;
        auto shape                   = outputs[0]->GetBlobDesc().dims;
        desc.dims                    = concat_inputs_[0]->GetBlobDesc().dims;
        int out_concat_dim_size      = 0;
        const int concat_inputs_size = concat_inputs_.size();
        for (int idx = 0; idx < concat_inputs_size; idx++) {
            out_concat_dim_size += concat_inputs_[idx]->GetBlobDesc().dims[axis_];
        }
        desc.dims[axis_] = out_concat_dim_size;

        concat_output_blob_ = std::make_shared<Blob>(desc);
        if (concat_output_blob_ == nullptr) {
            LOGE("Create reshape output blob in Concat failed!\n");
            return Status(TNNERR_CREATE_LAYER, "Create reshape output blob in Concat failed!");
        }
        concat_outputs_.push_back(concat_output_blob_.get());

        OpenCLRuntime *opencl_runtime = OpenCLRuntime::GetInstance();
        int climage_w = UP_DIV(DimsFunctionUtils::GetDim(desc.dims, 1), 4) * DimsFunctionUtils::GetDim(desc.dims, 3);
        int climage_h = DimsFunctionUtils::GetDim(desc.dims, 0) * DimsFunctionUtils::GetDim(desc.dims, 2);
        DimsVector imageshape{climage_w, climage_h};
        cl_channel_type data_type = CL_FLOAT;
        if (opencl_runtime->GetPrecision() != PRECISION_HIGH)
            data_type = CL_HALF_FLOAT;
        cl_int err           = CL_SUCCESS;
        concat_output_image_ = std::make_shared<cl::Image2D>(*opencl_runtime->Context(), CL_MEM_READ_WRITE,
                                                             cl::ImageFormat(CL_RGBA, data_type), imageshape[0],
                                                             imageshape[1], 0, nullptr, &err);
        if (err != CL_SUCCESS) {
            CHECK_CL_SUCCESS(err)
            return Status(TNNERR_OPENCL_MEMALLOC_ERROR, "OpenCL malloc memory failed");
        }

        BlobHandle blob_handle;
        blob_handle.base = concat_output_image_.get();
        concat_output_blob_->SetHandle(blob_handle);

        // Init LayerAcc
        std::shared_ptr<ReshapeLayerParam> reshape_param = std::make_shared<ReshapeLayerParam>();
        reshape_param->type         = "Reshape";
        reshape_param->name         = layer_name_ + "_Output_Reshape";
        reshape_param->reshape_type = 0;
        reshape_param->axis         = 0;
        reshape_param->num_axes     = shape.size();
        reshape_param->shape        = shape;
        output_reshape_layer_acc_->Init(ocl_context_, reshape_param.get(), nullptr, concat_outputs_, outputs);
        reshape_param_vec_.emplace_back(reshape_param);
    }

    return ret;
}

DimsVector OpenCLConcatLayerAcc::CalculateShape(const DimsVector &dims) {
    ConcatLayerParam *concat_param = dynamic_cast<ConcatLayerParam *>(param_);

    const int axis = concat_param->axis;

    if (axis == 0) {
        return {DimsFunctionUtils::GetDim(dims, 0), DimsVectorUtils::Count(dims, 1), 1, 1};
    }

    DimsVector target_dims = {DimsFunctionUtils::GetDim(dims, 0), 1, DimsFunctionUtils::GetDim(dims, axis), 1};
    if (axis > 1) {
        target_dims[1] = DimsVectorUtils::Count(dims, 1, axis);
    }
    if (axis < dims.size() - 1) {
        target_dims[3] = DimsVectorUtils::Count(dims, axis + 1);
    }

    return target_dims;
}

int OpenCLConcatLayerAcc::CalculateAxis(int axis, int dims_size) {
    if (dims_size <= 4 || axis == 0) {
        return axis;
    }

    return 2;
}

REGISTER_OPENCL_ACC(Concat, LAYER_CONCAT)
REGISTER_OPENCL_LAYOUT(LAYER_CONCAT, DATA_FORMAT_NHC4W4);

}  // namespace TNN_NS
