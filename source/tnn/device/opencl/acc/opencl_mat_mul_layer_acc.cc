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

#include "tnn/device/opencl/acc/opencl_mat_mul_layer_acc.h"
#include "tnn/device/opencl/imagebuffer_convertor.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

Status OpenCLMatMulLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                        const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init MatMul Acc\n");
    Status ret = OpenCLLayerAcc::Init(context, param, resource, inputs, outputs);
    CHECK_TNN_OK(ret)

    run_3d_ndrange_ = false;
    op_name_        = "MatMul";

    MatMulLayerParam *matmul_param = dynamic_cast<MatMulLayerParam *>(param);
    CHECK_PARAM_NULL(matmul_param);

    matrix_a_dims_ = matmul_param->matrix_a_dims;
    matrix_b_dims_ = matmul_param->matrix_b_dims;
    if (matrix_a_dims_.size() == 1) {
        matrix_a_dims_.insert(matrix_a_dims_.begin(), 1);
    }
    if (matrix_b_dims_.size() == 1) {
        matrix_b_dims_.push_back(1);
    }

    matrix_c_dims_       = outputs[0]->GetBlobDesc().dims;
    int M         = matrix_a_dims_[matrix_a_dims_.size() - 2];
    int K         = matrix_a_dims_[matrix_a_dims_.size() - 1];
    int N         = matrix_b_dims_[matrix_b_dims_.size() - 1];
    int count_a   = DimsVectorUtils::Count(matrix_a_dims_);
    int count_b   = DimsVectorUtils::Count(matrix_b_dims_);
    int count_c   = DimsVectorUtils::Count(matrix_c_dims_);
    int batch_a   = count_a / (M * K);
    int batch_b   = count_b / (K * N);
    int batch_c   = count_c / (M * N);

    // input0, input1, output
    reshape_inputs_.resize(3);
    reshape_outputs_.resize(3);

    // Load weights from layer resource
    if (inputs.size() == 1) {
        MatMulLayerResource *matmul_resource = dynamic_cast<MatMulLayerResource *>(resource);
        CHECK_PARAM_NULL(matmul_resource);
        RawBuffer &weight_handle = matmul_resource->weight;
        DataType data_type       = weight_handle.GetDataType();
        weight_position_         = matmul_param->weight_position; 
        // get weights
        int weights_height = batch_b * K;
        int weights_width  = N;
        if (weight_position_ == 0) {
            weights_height = batch_a * M;
            weights_width  = K;
        }
        if (weight_handle.GetDataType() == DATA_TYPE_FLOAT) {
            // get float pointer from raw buffer.
            float *weights_data_ptr = weight_handle.force_to<float *>();
            if (weights_data_ptr == nullptr) {
                return Status(TNNERR_OPENCL_ACC_INIT_ERROR, "pointer is null");
            }
            ret = ConvertWeights(weights_data_ptr, weights_width, weights_height);
            CHECK_TNN_OK(ret)
        } else {
            // if handle is half, need convert to float first.
            auto float_data_ptr = GetFloatFromRawBuffer(weight_handle);
            if (float_data_ptr == nullptr) {
                return Status(TNNERR_OPENCL_ACC_INIT_ERROR, "pointer is null");
            }
            ret = ConvertWeights(float_data_ptr.get(), weights_width, weights_height);
            CHECK_TNN_OK(ret)
        }
    }

    // create kernel
    std::string kernel_name = "MatMul";
    if (outputs[0]->GetBlobDesc().dims.size() == 6) {
        kernel_name = "MatMul6D";
    }
    ret = CreateExecuteUnit(execute_units_[0], "matmul", kernel_name, build_options_);
    if (ret != TNN_OK) {
        LOGE("create execute unit failed!\n");
        return ret;
    }

    return TNN_OK;
}

OpenCLMatMulLayerAcc::~OpenCLMatMulLayerAcc() {}

Status OpenCLMatMulLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("MatMul Layer Reshape\n");
    Status ret = OpenCLLayerAcc::Reshape(inputs, outputs);
    CHECK_TNN_OK(ret)

    auto input0_dims    = inputs[0]->GetBlobDesc().dims;
    auto output_dims    = outputs[0]->GetBlobDesc().dims;

    if (output_dims.size() == 6) {
        need_reshape_ = {false, false, false};

        DimsVector input1_dims;
        auto idx = SetExecuteUnit2DSizeInfoDefault(execute_units_[0], output_dims);
        if (inputs.size() == 2) {
            input1_dims = inputs[1]->GetBlobDesc().dims;
            execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)inputs[0]->GetHandle().base));
            execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)inputs[1]->GetHandle().base));
        } else {
            if (weight_position_ == 1) {
                input1_dims = reshape_outputs_[1][0]->GetBlobDesc().dims;
                execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)inputs[0]->GetHandle().base));
                execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)reshape_outputs_[1][0]->GetHandle().base));
            } else {
                input0_dims = reshape_outputs_[0][0]->GetBlobDesc().dims;
                input1_dims = inputs[0]->GetBlobDesc().dims;
                execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)reshape_outputs_[0][0]->GetHandle().base));
                execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)inputs[0]->GetHandle().base));
            }
        }

        execute_units_[0].ocl_kernel.setArg(idx++, input0_dims.size() * sizeof(int), input0_dims.data());
        execute_units_[0].ocl_kernel.setArg(idx++, input1_dims.size() * sizeof(int), input1_dims.data());
        execute_units_[0].ocl_kernel.setArg(idx++, output_dims.size() * sizeof(int), output_dims.data());
        execute_units_[0].ocl_kernel.setArg(idx++, UP_DIV(input0_dims[1], 4));
        execute_units_[0].ocl_kernel.setArg(idx++, UP_DIV(input1_dims[1], 4));
        execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)outputs[0]->GetHandle().base));

        return TNN_OK;
    }

    if (inputs.size() == 2) {
        bool need_reshape = false;
        ret = InitReshapeLayer(inputs[0], reshape_layer_acc_[0], need_reshape, reshape_inputs_[0],
                               reshape_outputs_[0], reshape_blob_[0], 0);
        CHECK_TNN_OK(ret)
        need_reshape_[0] = need_reshape;

        ret = InitReshapeLayer(inputs[1], reshape_layer_acc_[1], need_reshape, reshape_inputs_[1],
                               reshape_outputs_[1], reshape_blob_[1], 1);
        CHECK_TNN_OK(ret)
        need_reshape_[1] = need_reshape;

        ret = InitReshapeLayer(outputs[0], reshape_layer_acc_[2], need_reshape, reshape_inputs_[2],
                               reshape_outputs_[2], reshape_blob_[2], 2);
        CHECK_TNN_OK(ret)
        need_reshape_[2] = need_reshape;
    } else {
        int input0_position = weight_position_ == 1 ? 0 : 1;
        bool need_reshape = false;
        ret = InitReshapeLayer(inputs[0], reshape_layer_acc_[input0_position],
                               need_reshape, reshape_inputs_[input0_position],
                               reshape_outputs_[input0_position], reshape_blob_[input0_position],
                               input0_position);
        CHECK_TNN_OK(ret)
        need_reshape_[input0_position] = need_reshape;

        ret = InitReshapeLayer(outputs[0], reshape_layer_acc_[2], need_reshape, reshape_inputs_[2],
                               reshape_outputs_[2], reshape_blob_[2], 2);
        CHECK_TNN_OK(ret)
        need_reshape_[2] = need_reshape;
    }

    // reshape
    for (int i = 0; i < 3; i++) {
        if (need_reshape_[i]) {
            if (reshape_layer_acc_[i] == nullptr) {
                return Status(TNNERR_OPENCL_ACC_RESHAPE_ERROR, "reshape layer acc in MatMul is null");
            }
            ret = reshape_layer_acc_[i]->Reshape(reshape_inputs_[i], reshape_outputs_[i]);
            CHECK_TNN_OK(ret)
        }
    }

    int M         = matrix_a_dims_[matrix_a_dims_.size() - 2];
    int K         = matrix_a_dims_[matrix_a_dims_.size() - 1];
    int N         = matrix_b_dims_[matrix_b_dims_.size() - 1];
    int count_a   = DimsVectorUtils::Count(matrix_a_dims_);
    int count_b   = DimsVectorUtils::Count(matrix_b_dims_);
    int count_c   = DimsVectorUtils::Count(matrix_c_dims_);
    int batch_a   = count_a / (M * K);
    int batch_b   = count_b / (K * N);
    int batch_c   = count_c / (M * N);

    const int K_blocks = UP_DIV(K, 4);
    const int K_remain = K % 4;

    execute_units_[0].global_work_size = {static_cast<uint32_t>(UP_DIV(N, 4)), static_cast<uint32_t>(batch_c * M)};
    execute_units_[0].local_work_size  = {64, 1};

    uint32_t idx = 0;
    execute_units_[0].ocl_kernel.setArg(idx++, execute_units_[0].global_work_size[0]);
    execute_units_[0].ocl_kernel.setArg(idx++, execute_units_[0].global_work_size[1]);
    if (inputs.size() == 2) {
        if (need_reshape_[0]) {
            execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)reshape_outputs_[0][0]->GetHandle().base));
        } else {
            execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)inputs[0]->GetHandle().base));
        }
        if (need_reshape_[1]) {
            execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)reshape_outputs_[1][0]->GetHandle().base));
        } else {
            execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)inputs[1]->GetHandle().base));
        }
    } else {
        if (weight_position_ == 1) {
            if (need_reshape_[0]) {
                execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)reshape_outputs_[0][0]->GetHandle().base));
            } else {
                execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)inputs[0]->GetHandle().base));
            }
            // get weight from reshape output
            execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)reshape_outputs_[1][0]->GetHandle().base));
        } else {
            // get weight from reshape output
            execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)reshape_outputs_[0][0]->GetHandle().base));
            if (need_reshape_[1]) {
                execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)reshape_outputs_[1][0]->GetHandle().base));
            } else {
                execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)inputs[0]->GetHandle().base));
            }
        }
    }

    execute_units_[0].ocl_kernel.setArg(idx++, M);
    execute_units_[0].ocl_kernel.setArg(idx++, K_blocks);
    execute_units_[0].ocl_kernel.setArg(idx++, K);
    execute_units_[0].ocl_kernel.setArg(idx++, K_remain);
    execute_units_[0].ocl_kernel.setArg(idx++, batch_a);
    execute_units_[0].ocl_kernel.setArg(idx++, batch_b);
    if (need_reshape_[2]) {
        execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)reshape_inputs_[2][0]->GetHandle().base));
    } else {
        execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)outputs[0]->GetHandle().base));
    }

    return TNN_OK;
}

Status OpenCLMatMulLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    Status ret = TNN_OK;
    for (int i = 0; i < 2; i++) {
        if (need_reshape_[i]) {
            // reshape first
            if (reshape_layer_acc_[i] == nullptr) {
                return Status(TNNERR_OPENCL_ACC_FORWARD_ERROR, "reshape layer acc in MatMul is null");
            }
            ret = reshape_layer_acc_[i]->Forward(reshape_inputs_[i], reshape_outputs_[i]);
            CHECK_TNN_OK(ret)
        }
    }

    ret = OpenCLLayerAcc::Forward(inputs, outputs);
    CHECK_TNN_OK(ret)

    if (need_reshape_[2]) {
        if (reshape_layer_acc_[2] == nullptr) {
            return Status(TNNERR_OPENCL_ACC_FORWARD_ERROR, "reshape layer acc in MatMul is null");
        }
        ret = reshape_layer_acc_[2]->Forward(reshape_inputs_[2], reshape_outputs_[2]);
        CHECK_TNN_OK(ret)
    }

    return TNN_OK;
}

Status OpenCLMatMulLayerAcc::InitReshapeLayer(
        Blob *blob,
        std::shared_ptr<OpenCLReshapeLayerAcc>& layer,
        bool& need_reshape,
        std::vector<Blob *> &reshape_layer_inputs,
        std::vector<Blob *> &reshape_layer_outputs,
        std::shared_ptr<Blob> &reshape_blob,
        int position) {
    Status ret = TNN_OK;
    auto dims = blob->GetBlobDesc().dims;
    if ((position != 0 && dims.size() <= 2) || (dims.size() == 2)) {
        need_reshape = false;
        return TNN_OK;
    }

    need_reshape = true;
    reshape_layer_inputs.clear();
    if (position != 2) {
        reshape_layer_inputs.push_back(blob);
    } else {
        BlobDesc input_desc     = blob->GetBlobDesc();
        auto dims               = blob->GetBlobDesc().dims;
        int dim0 = 1, dim1 = 1, dim2 = 1, dim3 = 1;
        dim1 = dims.back();
        dim0 = DimsVectorUtils::Count(dims) / dim1;
        input_desc.dims = {dim0, dim1, dim2, dim3};
        reshape_blob    = std::make_shared<Blob>(input_desc, true);
        if (reshape_blob == nullptr) {
            LOGE("Create reshape input blob in MatMul failed!\n");
            return Status(TNNERR_CREATE_LAYER, "Create reshape input blob in MatMul failed!");
        }
        reshape_layer_inputs.push_back(reshape_blob.get());
    }

    layer = std::make_shared<OpenCLReshapeLayerAcc>();
    if (layer == nullptr) {
        LOGE("Create Reshape Layer Acc in MatMul failed!\n");
        return Status(TNNERR_CREATE_LAYER, "Create Reshape Layer Acc in MatMul failed!");
    }

    // create output_blob
    BlobDesc output_desc    = blob->GetBlobDesc();
    output_desc.data_format = DATA_FORMAT_NHC4W4;
    if (position != 2) {
        auto dims               = blob->GetBlobDesc().dims;
        int dim0 = 1, dim1 = 1, dim2 = 1, dim3 = 1;
        if (position == 0 && dims.size() == 1) {
            dim1 = dims[0];
        } else {
            dim1 = dims.back();
            dim0 = DimsVectorUtils::Count(dims) / dim1;
        }
        output_desc.dims = {dim0, dim1, dim2, dim3};
        reshape_blob    = std::make_shared<Blob>(output_desc, true);
        if (reshape_blob == nullptr) {
            LOGE("Create reshape output blob in MatMul failed!\n");
            return Status(TNNERR_CREATE_LAYER, "Create reshape output blob in MatMul failed!");
        }
        reshape_layer_outputs.clear();
        reshape_layer_outputs.push_back(reshape_blob.get());
    } else {
        reshape_layer_outputs.clear();
        reshape_layer_outputs.push_back(blob);
    }    

    // Init LayerAcc
    std::shared_ptr<ReshapeLayerParam> reshape_param = std::make_shared<ReshapeLayerParam>();
    if (position != 2) {
        reshape_param->name         = "MatMul_Reshape";
        reshape_param->reshape_type = 0;
        reshape_param->axis         = 0;
        reshape_param->num_axes     = 4;
        reshape_param->shape        = output_desc.dims;
        layer->Init(ocl_context_, reshape_param.get(), nullptr, reshape_layer_inputs, reshape_layer_outputs);
    } else {
        reshape_param->name         = "MatMul_Reshape";
        reshape_param->reshape_type = 0;
        reshape_param->axis         = 0;
        reshape_param->num_axes     = blob->GetBlobDesc().dims.size();
        reshape_param->shape        = blob->GetBlobDesc().dims;
        layer->Init(ocl_context_, reshape_param.get(), nullptr, reshape_layer_inputs, reshape_layer_outputs);
    }
    reshape_param_vec_.emplace_back(reshape_param);

    return ret;
}

Status OpenCLMatMulLayerAcc::ConvertWeights(float *weights_data_ptr, int weight_w, int weight_h) {
    OpenCLRuntime *opencl_runtime = OpenCLRuntime::GetInstance();

    // copy weights data into clBuffer
    DimsVector weight_shape{weight_h, weight_w, 1, 1};
    shared_ptr<OpenCLMemory> weight_buffer(new OpenCLMemory(TNN_CL_BUFFER));
    cl_int ret = CL_SUCCESS;
    cl::Buffer buffer(*opencl_runtime->Context(), CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                      DimsVectorUtils::Count(weight_shape) * sizeof(float), nullptr, &ret);
    if (ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(ret)
        return Status(TNNERR_OPENCL_MEMALLOC_ERROR, "OpenCL malloc memory failed");
    }
    weight_buffer->SetData(&buffer);
    ret = ocl_context_->CommandQueue()->enqueueWriteBuffer(
        buffer, CL_TRUE, 0, DimsVectorUtils::Count(weight_shape) * sizeof(float), weights_data_ptr);
    if (ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(ret)
        return Status(TNNERR_OPENCL_RUNTIME_ERROR, "OpenCL enqueueWriteBuffer failed");
    }

    // create weights blob
    BlobDesc desc;
    desc.device_type = DEVICE_OPENCL;
    desc.data_type = opencl_runtime->GetPrecision() == PRECISION_HIGH ? DATA_TYPE_FLOAT : DATA_TYPE_HALF;
    desc.dims = weight_shape;
    desc.data_format = DATA_FORMAT_NHC4W4;
    reshape_blob_[weight_position_] = std::make_shared<Blob>(desc, true);
    reshape_outputs_[weight_position_].clear();
    reshape_outputs_[weight_position_].push_back(reshape_blob_[weight_position_].get());

    // transfer from clBuffer to clImage
    ImageBufferConvertor convertor(opencl_runtime, ocl_context_->CommandQueue());
    std::shared_ptr<OpenCLMemory> weight_memory;
    weight_memory.reset(new OpenCLMemory(TNN_CL_IMAGE));
    weight_memory->SetData(reshape_blob_[weight_position_]->GetHandle().base, false);
    Status ret_convert = convertor.ConvertBufferToImage(
            weight_buffer.get(), NHWC_BUFFER, weight_shape, weight_memory.get(), true);
    CHECK_TNN_OK(ret_convert)

    return TNN_OK;
}

REGISTER_OPENCL_ACC(MatMul, LAYER_MATMUL)
REGISTER_OPENCL_LAYOUT(LAYER_MATMUL, DATA_FORMAT_NHC4W4);

}  // namespace TNN_NS
