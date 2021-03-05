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

#include "tnn/device/opencl/acc/opencl_lstm_layer_acc.h"
#include "tnn/device/opencl/imagebuffer_convertor.h"

namespace TNN_NS {
Status OpenCLLSTMONNXLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                    const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init LSTMONNX Acc\n");
    Status ret = OpenCLLayerAcc::Init(context, param, resource, inputs, outputs);
    CHECK_TNN_OK(ret)

    run_3d_ndrange_ = false;
    op_name_        = "LSTMONNX";

    execute_units_.resize(2);

    {
        std::string kernel_name = "LSTMONNXGates";
        ret                     = CreateExecuteUnit(execute_units_[0], "lstm", kernel_name);
        if (ret != TNN_OK) {
            LOGE("create execute unit failed!\n");
            return ret;
        }
    }

    {
        std::string kernel_name = "LSTMONNXForward";
        ret                     = CreateExecuteUnit(execute_units_[1], "lstm", kernel_name);
        if (ret != TNN_OK) {
            LOGE("create execute unit failed!\n");
            return ret;
        }
    }

    return TNN_OK;
}

OpenCLLSTMONNXLayerAcc::~OpenCLLSTMONNXLayerAcc() {}

Status OpenCLLSTMONNXLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("LSTMONNX Acc Reshape\n");
    auto layer_param = dynamic_cast<LSTMONNXLayerParam *>(param_);
    if (!layer_param) {
        LOGE("Error: LSTMONNX layer param is null\n");
        return Status(TNNERR_MODEL_ERR, "Error: LSTMONNX layer param is null");
    }    

    auto input          = inputs[0];
    Blob *blob_w        = inputs[1];
    Blob *blob_r        = inputs[2];
    Blob *blob_b        = inputs[3];
    Blob *blob_h0       = nullptr;
    Blob *blob_c0       = nullptr;
    auto output         = outputs[0];
    auto output_hidden  = outputs[1];
    auto output_cell    = outputs[2];

    if (inputs.size() >= 6) {
        blob_h0 = inputs[4];
        blob_c0 = inputs[5];
    } else {
        LOGE("TODO: empty initial states need to support");
        return Status(TNNERR_LAYER_ERR, "TODO: empty initial states need to support");
    }

    auto input_dims  = input->GetBlobDesc().dims;
    auto output_dims = output->GetBlobDesc().dims;
    const auto sequence = input_dims[0];
    const auto batch    = input_dims[1];
    const int input_size = input_dims[2];
    const int input_size_updiv_4 = UP_DIV(input_size, 4);
    const int hidden_size = output_dims[2];
    const int hidden_size_updiv_4 = UP_DIV(hidden_size, 4);
    int num_directions  = layer_param->direction >=2 ? 2 : 1;
    int reverse         = layer_param->direction == 1;

    OpenCLRuntime *opencl_runtime = OpenCLRuntime::GetInstance();
    int type_size = sizeof(float);
    if (opencl_runtime->GetPrecision() != PRECISION_HIGH) {
        type_size = 2;
    }

    {
        execute_units_[0].global_work_size = {static_cast<uint32_t>(hidden_size * num_directions),
                                              static_cast<uint32_t>(sequence * batch)};
        execute_units_[0].local_work_size = LocalWS2DDefault(execute_units_[0]);
        uint32_t idx = 0;
        execute_units_[0].ocl_kernel.setArg(idx++, execute_units_[0].global_work_size[0]);
        execute_units_[0].ocl_kernel.setArg(idx++, execute_units_[0].global_work_size[1]);
        execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)input->GetHandle().base));
        execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)blob_w->GetHandle().base));
        execute_units_[0].ocl_kernel.setArg(idx++, input_size_updiv_4);
        execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)ocl_gates_->GetData()));
    }

    {
        execute_units_[1].global_work_size = {static_cast<uint32_t>(hidden_size_updiv_4 * num_directions),
                                              static_cast<uint32_t>(batch)};
        execute_units_[1].local_work_size = LocalWS2DDefault(execute_units_[0]);
        uint32_t idx = 0;
        execute_units_[1].ocl_kernel.setArg(idx++, execute_units_[0].global_work_size[0]);
        execute_units_[1].ocl_kernel.setArg(idx++, execute_units_[0].global_work_size[1]);
        execute_units_[1].ocl_kernel.setArg(idx++, *((cl::Image *)ocl_gates_->GetData()));
        execute_units_[1].ocl_kernel.setArg(idx++, *((cl::Image *)blob_r->GetHandle().base));
        execute_units_[1].ocl_kernel.setArg(idx++, *((cl::Image *)blob_b->GetHandle().base));
        execute_units_[1].ocl_kernel.setArg(idx++, *((cl::Image *)blob_h0->GetHandle().base));
        execute_units_[1].ocl_kernel.setArg(idx++, *((cl::Image *)blob_c0->GetHandle().base));
        execute_units_[1].ocl_kernel.setArg(idx++, sequence);
        execute_units_[1].ocl_kernel.setArg(idx++, num_directions);
        execute_units_[1].ocl_kernel.setArg(idx++, hidden_size_updiv_4);
        execute_units_[1].ocl_kernel.setArg(idx++, reverse);
        int h_local_size = batch * num_directions * hidden_size_updiv_4 * 4 * type_size;
        execute_units_[1].ocl_kernel.setArg(idx++, h_local_size, nullptr);
        execute_units_[1].ocl_kernel.setArg(idx++, *((cl::Image *)output->GetHandle().base));
        execute_units_[1].ocl_kernel.setArg(idx++, *((cl::Image *)output_hidden->GetHandle().base));
        execute_units_[1].ocl_kernel.setArg(idx++, *((cl::Image *)output_cell->GetHandle().base));
    }

    return TNN_OK;
}

Status OpenCLLSTMONNXLayerAcc::ReloadConstantBlobs(const std::vector<Blob *> &inputs) {
    auto const_resource = const_resource_;

    if (inputs.size() < 4) {
        return Status(TNNERR_LAYER_ERR, "LSTM ONNX has invalid inputs");
    }

    // load w from constant blobs
    auto w = inputs[2];
    std::string name = w->GetBlobDesc().name;
    if (const_resource == nullptr || const_resource->find(name) == const_resource->end()) {
        return Status(TNNERR_LAYER_ERR, "LSTM ONNX has invalid input-w");
    }

    auto buffer = (*const_resource)[name];
    std::shared_ptr<Blob> blob = nullptr;
    if (const_blob_map_.find(name) != const_blob_map_.end()) {
        blob = const_blob_map_[name];
    } else {
        auto status = ConvertWeights(buffer, blob);
        RETURN_ON_NEQ(status, TNN_OK);
        blob->flag = DATA_FLAG_CHANGE_NEVER;
        const_blob_map_[name] = blob;
    }
    w->SetHandle(blob->GetHandle());

    // load r from constant blobs
    auto r = inputs[3];
    name = r->GetBlobDesc().name;
    if (const_resource == nullptr || const_resource->find(name) == const_resource->end()) {
        return Status(TNNERR_LAYER_ERR, "LSTM ONNX has invalid input-r");
    }

    buffer = (*const_resource)[name];
    blob = nullptr;
    if (const_blob_map_.find(name) != const_blob_map_.end()) {
        blob = const_blob_map_[name];
    } else {
        auto status = ConvertWeights(buffer, blob);
        RETURN_ON_NEQ(status, TNN_OK);
        blob->flag = DATA_FLAG_CHANGE_NEVER;
        const_blob_map_[name] = blob;
    }
    r->SetHandle(blob->GetHandle());

    // load b from constant blobs
    auto b = inputs[4];
    name = b->GetBlobDesc().name;
    if (const_resource == nullptr || const_resource->find(name) == const_resource->end()) {
        return Status(TNNERR_LAYER_ERR, "LSTM ONNX has invalid input-r");
    }

    buffer = (*const_resource)[name];
    blob = nullptr;
    if (const_blob_map_.find(name) != const_blob_map_.end()) {
        blob = const_blob_map_[name];
    } else {
        auto status = ConvertBias(buffer, blob);
        RETURN_ON_NEQ(status, TNN_OK);
        blob->flag = DATA_FLAG_CHANGE_NEVER;
        const_blob_map_[name] = blob;
    }
    b->SetHandle(blob->GetHandle());

    return TNN_OK;
}

Status OpenCLLSTMONNXLayerAcc::ConvertWeights(std::shared_ptr<RawBuffer> buffer, std::shared_ptr<Blob>& blob) {
    if (!buffer || buffer->GetBufferDims().size() != 3) {
        return Status(TNNERR_PARAM_ERR, "weights buffer is invalid");
    }

    float *weights_data_ptr;
    if (buffer->GetDataType() == DATA_TYPE_FLOAT) {
        // get float pointer from raw buffer
        weights_data_ptr = buffer->force_to<float *>();
        if (weights_data_ptr == nullptr) {
            return Status(TNNERR_OPENCL_ACC_INIT_ERROR, "pointer is null");
        }
    } else {
        // if handle is half, need convert to float first.
        auto float_data_ptr = GetFloatFromRawBuffer(*buffer);
        if (float_data_ptr == nullptr) {
            return Status(TNNERR_OPENCL_ACC_INIT_ERROR, "pointer is null");
        }
        weights_data_ptr = float_data_ptr.get();
    }

    OpenCLRuntime *opencl_runtime = OpenCLRuntime::GetInstance();
    int num_directions = buffer->GetBufferDims()[0];
    int gates_hidden_size = buffer->GetBufferDims()[1];
    int weights_width = buffer->GetBufferDims()[2];
    int weights_w = num_directions * gates_hidden_size, weights_h = weights_width;

    // weights: [num_directions, 4 * hidden_size, weights_width]
    // transpose
    DimsVector weights_shape = {weights_h, weights_w, 1, 1};
    std::shared_ptr<float> weights_data_ptr_trans(new float[weights_w * weights_h]);
    for (size_t i = 0; i < weights_h; i++) {
        for (size_t j = 0; j < weights_w; j++) {
            weights_data_ptr_trans.get()[j + i * weights_w] = weights_data_ptr[i + j * weights_h];
        }
    }

    // transposed weights: [weights_width, 4*hidden_size*num_directions]
    // copy weights data into clBuffer
    std::shared_ptr<OpenCLMemory> weights_buffer(new OpenCLMemory(TNN_CL_BUFFER));
    cl_int ret = CL_SUCCESS;
    cl::Buffer cl_buffer(*opencl_runtime->Context(), CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                         DimsVectorUtils::Count(weights_shape) * sizeof(float), nullptr, &ret);
    if (ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(ret)
        return Status(TNNERR_OPENCL_MEMALLOC_ERROR, "OpenCL malloc memory falied");
    }
    weights_buffer->SetData(&cl_buffer);
    ret = ocl_context_->CommandQueue()->enqueueWriteBuffer(cl_buffer, CL_TRUE, 0,
            DimsVectorUtils::Count(weights_shape) * sizeof(float), weights_data_ptr_trans.get());
    if (ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(ret)
        return Status(TNNERR_OPENCL_MEMUNMAP_ERROR, "OpenCL enqueueWriteBuffer falied");
    }

    BlobDesc desc;
    // use CNH4 format to desc weights blob
    DimsVector weights_cnh4_shape = {1, weights_h, weights_w};
    desc.device_type = DEVICE_OPENCL;
    desc.data_type = opencl_runtime->GetPrecision() == PRECISION_HIGH ? DATA_TYPE_FLOAT : DATA_TYPE_HALF;
    desc.dims = weights_cnh4_shape;
    desc.data_format = DATA_FORMAT_CNH4;
    if (buffer->GetBytesSize() > 0) {
        blob = std::make_shared<Blob>(desc, true);
    } else {
        return Status(TNNERR_PARAM_ERR, "weights buffer is empty");
    }

    // transfer from clBuffer to clImage
    ImageBufferConvertor convertor(opencl_runtime, ocl_context_->CommandQueue());
    std::shared_ptr<OpenCLMemory> weights_memory;
    weights_memory.reset(new OpenCLMemory(TNN_CL_IMAGE));
    weights_memory->SetData(blob->GetHandle().base, false);
    Status ret_convert = convertor.ConvertBufferToImage(
            weights_buffer.get(), NHWC_BUFFER, weights_shape, weights_memory.get(), true);
    CHECK_TNN_OK(ret_convert)

    return TNN_OK;
}

Status OpenCLLSTMONNXLayerAcc::ConvertBias(std::shared_ptr<RawBuffer> buffer, std::shared_ptr<Blob>& blob) {
    if (!buffer || buffer->GetBufferDims().size() != 2) {
        return Status(TNNERR_PARAM_ERR, "bias buffer is invalid");
    }

    float *bias_data_ptr;
    if (buffer->GetDataType() == DATA_TYPE_FLOAT) {
        // get float pointer from raw buffer
        bias_data_ptr = buffer->force_to<float *>();
        if (bias_data_ptr == nullptr) {
            return Status(TNNERR_OPENCL_ACC_INIT_ERROR, "pointer is null");
        }
    } else {
        // if handle is half, need convert to float first.
        auto float_data_ptr = GetFloatFromRawBuffer(*buffer);
        if (float_data_ptr == nullptr) {
            return Status(TNNERR_OPENCL_ACC_INIT_ERROR, "pointer is null");
        }
        bias_data_ptr = float_data_ptr.get();
    }

    OpenCLRuntime *opencl_runtime = OpenCLRuntime::GetInstance();
    int num_directions = buffer->GetBufferDims()[0];
    int gates_hidden_size = buffer->GetBufferDims()[1];
    int bias_w = num_directions, bias_h = gates_hidden_size;

    // bias: [num_directions, 8 * hidden_size]
    DimsVector bias_shape = {bias_w, bias_h, 1, 1};

    // copy bias data into clBuffer
    std::shared_ptr<OpenCLMemory> bias_buffer(new OpenCLMemory(TNN_CL_BUFFER));
    cl_int ret = CL_SUCCESS;
    cl::Buffer cl_buffer(*opencl_runtime->Context(), CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                         DimsVectorUtils::Count(bias_shape) * sizeof(float), nullptr, &ret);
    if (ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(ret)
        return Status(TNNERR_OPENCL_MEMALLOC_ERROR, "OpenCL malloc memory falied");
    }
    bias_buffer->SetData(&cl_buffer);
    ret = ocl_context_->CommandQueue()->enqueueWriteBuffer(cl_buffer, CL_TRUE, 0,
            DimsVectorUtils::Count(bias_shape) * sizeof(float), bias_data_ptr);
    if (ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(ret)
        return Status(TNNERR_OPENCL_MEMUNMAP_ERROR, "OpenCL enqueueWriteBuffer falied");
    }

    BlobDesc desc;
    // use CNH4 format to desc bias blob
    DimsVector bias_cnh4_shape = {1, bias_w, bias_h};
    desc.device_type = DEVICE_OPENCL;
    desc.data_type = opencl_runtime->GetPrecision() == PRECISION_HIGH ? DATA_TYPE_FLOAT : DATA_TYPE_HALF;
    desc.dims = bias_cnh4_shape;
    desc.data_format = DATA_FORMAT_CNH4;
    if (buffer->GetBytesSize() > 0) {
        blob = std::make_shared<Blob>(desc, true);
    } else {
        return Status(TNNERR_PARAM_ERR, "weights buffer is empty");
    }

    // transfer from clBuffer to clImage
    ImageBufferConvertor convertor(opencl_runtime, ocl_context_->CommandQueue());
    std::shared_ptr<OpenCLMemory> bias_memory;
    bias_memory.reset(new OpenCLMemory(TNN_CL_IMAGE));
    bias_memory->SetData(blob->GetHandle().base, false);
    Status ret_convert = convertor.ConvertBufferToImage(
            bias_buffer.get(), NHWC_BUFFER, bias_shape, bias_memory.get(), true);
    CHECK_TNN_OK(ret_convert)

    return TNN_OK;
}

std::vector<DataFormat> OpenCLLSTMONNXLayerAcc::SupportDataFormat(DataType data_type, int dims_size) {
    std::vector<DataFormat> support_list;
    if (dims_size == 3) {
        support_list.push_back(DATA_FORMAT_CNH4);
    }
    return support_list;
}

REGISTER_OPENCL_ACC(LSTMONNX, LAYER_LSTMONNX);
REGISTER_OPENCL_LAYOUT(LAYER_LSTMONNX, DATA_FORMAT_CNH4);
}  // namespace TNN_NS
