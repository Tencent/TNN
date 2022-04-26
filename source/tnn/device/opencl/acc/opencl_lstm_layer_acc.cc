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

    execute_units_.resize(3);

    {
        std::string kernel_name = "LSTMONNXGates";
        ret                     = CreateExecuteUnit(execute_units_[0], "lstm", kernel_name, build_options_);
        if (ret != TNN_OK) {
            LOGE("create execute unit failed!\n");
            return ret;
        }
    }

    {
        std::string kernel_name = "LSTMONNXForward";
        ret                     = CreateExecuteUnit(execute_units_[1], "lstm", kernel_name, build_options_);
        if (ret != TNN_OK) {
            LOGE("create execute unit failed!\n");
            return ret;
        }
    }

    {
        std::string kernel_name = "LSTMONNXResultConvert";
        ret                     = CreateExecuteUnit(execute_units_[2], "lstm", kernel_name, build_options_);
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
    Status ret = OpenCLLayerAcc::Reshape(inputs, outputs);
    CHECK_TNN_OK(ret)

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

    auto input_dims  = input->GetBlobDesc().dims;
    auto output_dims = output->GetBlobDesc().dims;
    const auto sequence = DimsFunctionUtils::GetDim(input_dims, 0);
    const auto batch    = DimsFunctionUtils::GetDim(input_dims, 1);
    const int input_size = DimsFunctionUtils::GetDim(input_dims, 2);
    const int input_size_updiv_4 = UP_DIV(input_size, 4);
    int num_directions  = layer_param->direction >=2 ? 2 : 1;
    const int hidden_size = DimsFunctionUtils::GetDim(output_dims, 2) / num_directions;
    const int hidden_size_updiv_4 = UP_DIV(hidden_size, 4);
    int reverse         = layer_param->direction == 1;

    if (inputs.size() >= 6) {
        blob_h0 = inputs[4];
        blob_c0 = inputs[5];
    } else {
        Status ret = CreateDefaultState(num_directions, batch, hidden_size, ocl_zero_state_blob_);
        if (ret != TNN_OK) {
            return Status(TNNERR_LAYER_ERR, "Empty initial states create failed");
        }
        blob_h0 = ocl_zero_state_blob_.get();
        blob_c0 = ocl_zero_state_blob_.get();
    }

    ret = AllocateTempBlob(num_directions, hidden_size, batch, sequence, ocl_gates_);
    if (ret != TNN_OK) {
        return Status(TNNERR_LAYER_ERR, "Allocate gates failed");
    }

    // special case, reverse output is stored after forward output
    bool need_temp_out = (hidden_size % 4 != 0 && num_directions == 2);
    if (need_temp_out) {
        ret = AllocateTempBlob(num_directions, hidden_size, batch, sequence, ocl_temp_out_);
        if (ret != TNN_OK) {
            return Status(TNNERR_LAYER_ERR, "Allocate gates failed");
        }
    }

    OpenCLRuntime *opencl_runtime = OpenCLRuntime::GetInstance();
    int type_size = sizeof(float);
    if (opencl_runtime->GetPrecision() != PRECISION_HIGH) {
        type_size = 2;
    }

    {
        execute_units_[0].global_work_size = {static_cast<uint32_t>(hidden_size_updiv_4 * 4 * num_directions),
                                              static_cast<uint32_t>(sequence * batch)};
        execute_units_[0].local_work_size = LocalWS2DDefault(execute_units_[0]);
        uint32_t idx = 0;
        execute_units_[0].ocl_kernel.setArg(idx++, execute_units_[0].global_work_size[0]);
        execute_units_[0].ocl_kernel.setArg(idx++, execute_units_[0].global_work_size[1]);
        execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)input->GetHandle().base));
        execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)blob_w->GetHandle().base));
        execute_units_[0].ocl_kernel.setArg(idx++, input_size_updiv_4);
        execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)ocl_gates_->GetHandle().base));
    }

    {
        execute_units_[1].global_work_size = {static_cast<uint32_t>(hidden_size_updiv_4 * num_directions),
                                              static_cast<uint32_t>(batch)};
        execute_units_[1].local_work_size = {static_cast<uint32_t>(hidden_size_updiv_4), 1};
        uint32_t idx = 0;
        execute_units_[1].ocl_kernel.setArg(idx++, execute_units_[1].global_work_size[0]);
        execute_units_[1].ocl_kernel.setArg(idx++, execute_units_[1].global_work_size[1]);
        execute_units_[1].ocl_kernel.setArg(idx++, *((cl::Image *)ocl_gates_->GetHandle().base));
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
        if (need_temp_out) {
            execute_units_[1].ocl_kernel.setArg(idx++, *((cl::Image *)ocl_temp_out_->GetHandle().base));
        } else {
            execute_units_[1].ocl_kernel.setArg(idx++, *((cl::Image *)output->GetHandle().base));
        }
        execute_units_[1].ocl_kernel.setArg(idx++, *((cl::Image *)output_hidden->GetHandle().base));
        execute_units_[1].ocl_kernel.setArg(idx++, *((cl::Image *)output_cell->GetHandle().base));
    }

    if (need_temp_out) {
        execute_units_[2].global_work_size = {
            static_cast<uint32_t>(UP_DIV(DimsFunctionUtils::GetDim(output_dims, 2), 4)),
            static_cast<uint32_t>(sequence * batch)};
        execute_units_[2].local_work_size = LocalWS2DDefault(execute_units_[2]);
        uint32_t idx = 0;
        execute_units_[2].ocl_kernel.setArg(idx++, execute_units_[2].global_work_size[0]);
        execute_units_[2].ocl_kernel.setArg(idx++, execute_units_[2].global_work_size[1]);
        execute_units_[2].ocl_kernel.setArg(idx++, *((cl::Image *)ocl_temp_out_->GetHandle().base));
        execute_units_[2].ocl_kernel.setArg(idx++, hidden_size);
        execute_units_[2].ocl_kernel.setArg(idx++, hidden_size_updiv_4);
        execute_units_[2].ocl_kernel.setArg(idx++, *((cl::Image *)output->GetHandle().base));
    } else {
        InsertUnactiveUnitId(2);
    }

    return TNN_OK;
}

Status OpenCLLSTMONNXLayerAcc::ReloadConstantBlobs(const std::vector<Blob *> &inputs, bool only_reload_shape_differ_blob) {
    auto const_resource = const_resource_;

    if (inputs.size() < 4) {
        return Status(TNNERR_LAYER_ERR, "LSTM ONNX has invalid inputs");
    }

    // load w from constant blobs
    auto w = inputs[1];
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
        blob->SetFlag(DATA_FLAG_CHANGE_NEVER);
        const_blob_map_[name] = blob;
    }
    w->SetHandle(blob->GetHandle());

    // load r from constant blobs
    auto r = inputs[2];
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
        blob->SetFlag(DATA_FLAG_CHANGE_NEVER);
        const_blob_map_[name] = blob;
    }
    r->SetHandle(blob->GetHandle());

    // load b from constant blobs
    auto b = inputs[3];
    name = b->GetBlobDesc().name;
    if (const_resource == nullptr || const_resource->find(name) == const_resource->end()) {
        return Status(TNNERR_LAYER_ERR, "LSTM ONNX has invalid input-b");
    }

    buffer = (*const_resource)[name];
    blob = nullptr;
    if (const_blob_map_.find(name) != const_blob_map_.end()) {
        blob = const_blob_map_[name];
    } else {
        auto status = ConvertBias(buffer, blob);
        RETURN_ON_NEQ(status, TNN_OK);
        blob->SetFlag(DATA_FLAG_CHANGE_NEVER);
        const_blob_map_[name] = blob;
    }
    b->SetHandle(blob->GetHandle());

    if (inputs.size() >= 6) {
        auto h0 = inputs[4];
        name = h0->GetBlobDesc().name;
        if (const_resource != nullptr && const_resource->find(name) != const_resource->end()) {
            buffer = (*const_resource)[name];
            blob = nullptr;
            if (const_blob_map_.find(name) != const_blob_map_.end()) {
                blob = const_blob_map_[name];
            } else {
                auto status = ConvertInitialState(buffer, blob);
                RETURN_ON_NEQ(status, TNN_OK);
                blob->SetFlag(DATA_FLAG_CHANGE_NEVER);
                const_blob_map_[name] = blob;
            }
            h0->SetHandle(blob->GetHandle());
        }

        auto c0 = inputs[5];
        name = c0->GetBlobDesc().name;
        if (const_resource != nullptr && const_resource->find(name) != const_resource->end()) {
            buffer = (*const_resource)[name];
            blob = nullptr;
            if (const_blob_map_.find(name) != const_blob_map_.end()) {
                blob = const_blob_map_[name];
            } else {
                auto status = ConvertInitialState(buffer, blob);
                RETURN_ON_NEQ(status, TNN_OK);
                blob->SetFlag(DATA_FLAG_CHANGE_NEVER);
                const_blob_map_[name] = blob;
            }
            c0->SetHandle(blob->GetHandle());
        }
    }

    return TNN_OK;
}

Status OpenCLLSTMONNXLayerAcc::ConvertWeights(std::shared_ptr<RawBuffer> buffer, std::shared_ptr<Blob>& blob) {
    if (!buffer || buffer->GetBufferDims().size() != 3) {
        return Status(TNNERR_PARAM_ERR, "weights buffer is invalid");
    }

    float *weights_data_ptr;
    std::shared_ptr<float> float_data_ptr;
    if (buffer->GetDataType() == DATA_TYPE_FLOAT) {
        // get float pointer from raw buffer
        weights_data_ptr = buffer->force_to<float *>();
        if (weights_data_ptr == nullptr) {
            return Status(TNNERR_OPENCL_ACC_INIT_ERROR, "pointer is null");
        }
    } else {
        // if handle is half, need convert to float first.
        float_data_ptr = GetFloatFromRawBuffer(*buffer);
        if (float_data_ptr == nullptr) {
            return Status(TNNERR_OPENCL_ACC_INIT_ERROR, "pointer is null");
        }
        weights_data_ptr = float_data_ptr.get();
    }

    // weights: [num_directions, 4 * hidden_size, weights_width]
    OpenCLRuntime *opencl_runtime = OpenCLRuntime::GetInstance();
    int num_directions = buffer->GetBufferDims()[0];
    int gates_hidden_size = buffer->GetBufferDims()[1];
    int weights_width = buffer->GetBufferDims()[2];
    int weights_w = num_directions * ALIGN_UP4(gates_hidden_size / 4) * 4, weights_h = weights_width;

    // copy weights data into clBuffer
    std::shared_ptr<OpenCLMemory> weights_buffer(new OpenCLMemory(TNN_CL_BUFFER));
    cl_int ret = CL_SUCCESS;
    cl::Buffer cl_buffer(*opencl_runtime->Context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                         DimsVectorUtils::Count(buffer->GetBufferDims()) * sizeof(float), nullptr, &ret);
    if (ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(ret)
        return Status(TNNERR_OPENCL_MEMALLOC_ERROR, "OpenCL malloc memory failed");
    }
    weights_buffer->SetData(&cl_buffer);
    ret = ocl_context_->CommandQueue()->enqueueWriteBuffer(cl_buffer, CL_TRUE, 0,
            DimsVectorUtils::Count(buffer->GetBufferDims()) * sizeof(float), weights_data_ptr);
    if (ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(ret)
        return Status(TNNERR_OPENCL_RUNTIME_ERROR, "OpenCL enqueueWriteBuffer failed");
    }

    BlobDesc desc;
    // use CNH4 format to desc weights blob
    DimsVector weights_image_shape = {1, weights_h, weights_w};
    desc.device_type = DEVICE_OPENCL;
    desc.data_type = opencl_runtime->GetPrecision() == PRECISION_HIGH ? DATA_TYPE_FLOAT : DATA_TYPE_HALF;
    desc.dims = weights_image_shape;
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
            weights_buffer.get(), LSTM_FILTER, buffer->GetBufferDims(), weights_memory.get(), true);
    CHECK_TNN_OK(ret_convert)

    return TNN_OK;
}

Status OpenCLLSTMONNXLayerAcc::ConvertBias(std::shared_ptr<RawBuffer> buffer, std::shared_ptr<Blob>& blob) {
    if (!buffer || buffer->GetBufferDims().size() != 2) {
        return Status(TNNERR_PARAM_ERR, "bias buffer is invalid");
    }

    float *bias_data_ptr;
    std::shared_ptr<float> float_data_ptr;
    if (buffer->GetDataType() == DATA_TYPE_FLOAT) {
        // get float pointer from raw buffer
        bias_data_ptr = buffer->force_to<float *>();
        if (bias_data_ptr == nullptr) {
            return Status(TNNERR_OPENCL_ACC_INIT_ERROR, "pointer is null");
        }
    } else {
        // if handle is half, need convert to float first.
        float_data_ptr = GetFloatFromRawBuffer(*buffer);
        if (float_data_ptr == nullptr) {
            return Status(TNNERR_OPENCL_ACC_INIT_ERROR, "pointer is null");
        }
        bias_data_ptr = float_data_ptr.get();
    }

    OpenCLRuntime *opencl_runtime = OpenCLRuntime::GetInstance();
    int num_directions = buffer->GetBufferDims()[0];
    int gates_hidden_size = buffer->GetBufferDims()[1];
    int hidden_size = gates_hidden_size / 8;
    int bias_w = ALIGN_UP4(hidden_size) * 8, bias_h = num_directions;

    // bias: [num_directions, 8 * hidden_size]
    // copy bias data into clBuffer
    std::shared_ptr<OpenCLMemory> bias_buffer(new OpenCLMemory(TNN_CL_BUFFER));
    cl_int ret = CL_SUCCESS;
    cl::Buffer cl_buffer(*opencl_runtime->Context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                         DimsVectorUtils::Count(buffer->GetBufferDims()) * sizeof(float), nullptr, &ret);
    if (ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(ret)
        return Status(TNNERR_OPENCL_MEMALLOC_ERROR, "OpenCL malloc memory failed");
    }
    bias_buffer->SetData(&cl_buffer);
    ret = ocl_context_->CommandQueue()->enqueueWriteBuffer(cl_buffer, CL_TRUE, 0,
            DimsVectorUtils::Count(buffer->GetBufferDims()) * sizeof(float), bias_data_ptr);
    if (ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(ret)
        return Status(TNNERR_OPENCL_RUNTIME_ERROR, "OpenCL enqueueWriteBuffer failed");
    }

    BlobDesc desc;
    // use CNH4 format to desc bias blob
    DimsVector bias_image_shape = {1, bias_h, bias_w};
    desc.device_type = DEVICE_OPENCL;
    desc.data_type = opencl_runtime->GetPrecision() == PRECISION_HIGH ? DATA_TYPE_FLOAT : DATA_TYPE_HALF;
    desc.dims = bias_image_shape;
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
            bias_buffer.get(), LSTM_BIAS, buffer->GetBufferDims(), bias_memory.get(), true);
    CHECK_TNN_OK(ret_convert)

    return TNN_OK;
}

Status OpenCLLSTMONNXLayerAcc::ConvertInitialState(std::shared_ptr<RawBuffer> buffer,
                                                   std::shared_ptr<Blob>& blob) {
    if (!buffer || buffer->GetBufferDims().size() != 3) {
        return Status(TNNERR_PARAM_ERR, "state buffer is invalid");
    }

    float *state_data_ptr;
    if (buffer->GetDataType() == DATA_TYPE_FLOAT) {
        // get float pointer from raw buffer
        state_data_ptr = buffer->force_to<float *>();
        if (state_data_ptr == nullptr) {
            return Status(TNNERR_OPENCL_ACC_INIT_ERROR, "pointer is null");
        }
    } else {
        // if handle is half, need convert to float first.
        auto float_data_ptr = GetFloatFromRawBuffer(*buffer);
        if (float_data_ptr == nullptr) {
            return Status(TNNERR_OPENCL_ACC_INIT_ERROR, "pointer is null");
        }
        state_data_ptr = float_data_ptr.get();
    }

    OpenCLRuntime *opencl_runtime = OpenCLRuntime::GetInstance();
    int num_directions = buffer->GetBufferDims()[0];
    int batch = buffer->GetBufferDims()[1];
    int hidden_size = buffer->GetBufferDims()[2];
    int state_w = num_directions * batch, state_h = hidden_size;

    // state: [num_directions, batch, hidden_size]
    // transpose
    DimsVector state_shape = {state_w, state_h, 1, 1};
    std::shared_ptr<float> state_data_ptr_trans(new float[state_w * state_h]);
    for (size_t d = 0; d < num_directions; d++) {
        for (size_t b = 0; b < batch; b++) {
            for (size_t i = 0; i < hidden_size; i++) {
                state_data_ptr_trans.get()[(b * num_directions + d) * hidden_size + i] =
                    state_data_ptr[(d * batch + b) * hidden_size + i];
            }
        }
    }

    // transposed state: [batch * num_directions, hidden_size]

    // copy state data into clBuffer
    std::shared_ptr<OpenCLMemory> state_buffer(new OpenCLMemory(TNN_CL_BUFFER));
    cl_int ret = CL_SUCCESS;
    cl::Buffer cl_buffer(*opencl_runtime->Context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                         DimsVectorUtils::Count(state_shape) * sizeof(float), nullptr, &ret);
    if (ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(ret)
        return Status(TNNERR_OPENCL_MEMALLOC_ERROR, "OpenCL malloc memory failed");
    }
    state_buffer->SetData(&cl_buffer);
    ret = ocl_context_->CommandQueue()->enqueueWriteBuffer(cl_buffer, CL_TRUE, 0,
            DimsVectorUtils::Count(state_shape) * sizeof(float), state_data_ptr_trans.get());
    if (ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(ret)
        return Status(TNNERR_OPENCL_API_ERROR, "OpenCL enqueueWriteBuffer failed");
    }

    BlobDesc desc;
    // use CNH4 format to desc state blob
    DimsVector state_cnh4_shape = {1, state_w, state_h};
    desc.device_type = DEVICE_OPENCL;
    desc.data_type = opencl_runtime->GetPrecision() == PRECISION_HIGH ? DATA_TYPE_FLOAT : DATA_TYPE_HALF;
    desc.dims = state_cnh4_shape;
    desc.data_format = DATA_FORMAT_CNH4;
    if (buffer->GetBytesSize() > 0) {
        blob = std::make_shared<Blob>(desc, true);
    } else {
        return Status(TNNERR_PARAM_ERR, "weights buffer is empty");
    }

    // transfer from clBuffer to clImage
    ImageBufferConvertor convertor(opencl_runtime, ocl_context_->CommandQueue());
    std::shared_ptr<OpenCLMemory> state_memory;
    state_memory.reset(new OpenCLMemory(TNN_CL_IMAGE));
    state_memory->SetData(blob->GetHandle().base, false);
    Status ret_convert = convertor.ConvertBufferToImage(
            state_buffer.get(), NHWC_BUFFER, state_shape, state_memory.get(), true);
    CHECK_TNN_OK(ret_convert)

    return TNN_OK;
}

Status OpenCLLSTMONNXLayerAcc::CreateDefaultState(int num_directions,
                                                  int batch,
                                                  int hidden_size,
                                                  std::shared_ptr<Blob>& blob) {
    DimsVector shape = {num_directions, batch, hidden_size};
    if (blob) {
        auto dims = blob->GetBlobDesc().dims;
        if (dims == shape) return TNN_OK;
    }
    OpenCLRuntime *opencl_runtime = OpenCLRuntime::GetInstance();
    BlobDesc desc;
    desc.device_type = DEVICE_OPENCL;
    desc.data_type = opencl_runtime->GetPrecision() == PRECISION_HIGH ? DATA_TYPE_FLOAT : DATA_TYPE_HALF;
    desc.dims = shape;
    desc.data_format = DATA_FORMAT_CNH4;
    blob = std::make_shared<Blob>(desc, true);
    int image_width = ALIGN_UP4(hidden_size), image_height = num_directions * batch;

    std::vector<float> zero_buffer_data(image_width * image_height, 0);
    std::shared_ptr<OpenCLMemory> state_buffer(new OpenCLMemory(TNN_CL_BUFFER));
    cl_int ret = CL_SUCCESS;
    cl::Buffer cl_buffer(*opencl_runtime->Context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                         image_width * image_height * sizeof(float), nullptr, &ret);
    if (ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(ret)
        return Status(TNNERR_OPENCL_MEMALLOC_ERROR, "OpenCL malloc memory failed");
    }
    state_buffer->SetData(&cl_buffer);
    ret = ocl_context_->CommandQueue()->enqueueWriteBuffer(cl_buffer, CL_TRUE, 0,
            image_width * image_height * sizeof(float), zero_buffer_data.data());
    if (ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(ret)
        return Status(TNNERR_OPENCL_API_ERROR, "OpenCL enqueueWriteBuffer failed");
    }

    // transfer from clBuffer to clImage
    ImageBufferConvertor convertor(opencl_runtime, ocl_context_->CommandQueue());
    std::shared_ptr<OpenCLMemory> state_memory;
    state_memory.reset(new OpenCLMemory(TNN_CL_IMAGE));
    state_memory->SetData(blob->GetHandle().base, false);
    DimsVector nhwc_buffer_shape = {image_height, image_width, 1, 1};
    Status ret_convert = convertor.ConvertBufferToImage(
            state_buffer.get(), NHWC_BUFFER, nhwc_buffer_shape, state_memory.get(), true);
    CHECK_TNN_OK(ret_convert)

    return TNN_OK;
}

// temp blob: [hidden_size * 4 * num_directions, sequence * batch]
Status OpenCLLSTMONNXLayerAcc::AllocateTempBlob(int num_directions,
                                                int hidden_size,
                                                int batch,
                                                int sequence,
                                                std::shared_ptr<Blob>& blob) {
    DimsVector shape = {sequence, batch, ALIGN_UP4(hidden_size) * 4 * num_directions};
    if (blob) {
        auto dims = blob->GetBlobDesc().dims;
        if (dims == shape) return TNN_OK;
    }
    OpenCLRuntime *opencl_runtime = OpenCLRuntime::GetInstance();
    BlobDesc desc;
    desc.device_type = DEVICE_OPENCL;
    desc.data_type = opencl_runtime->GetPrecision() == PRECISION_HIGH ? DATA_TYPE_FLOAT : DATA_TYPE_HALF;
    desc.dims = shape;
    desc.data_format = DATA_FORMAT_CNH4;
    blob = std::make_shared<Blob>(desc, true);

    std::vector<float> zero_buffer_data(DimsVectorUtils::Count(shape), 0);
    std::shared_ptr<OpenCLMemory> state_buffer(new OpenCLMemory(TNN_CL_BUFFER));
    cl_int ret = CL_SUCCESS;
    cl::Buffer cl_buffer(*opencl_runtime->Context(), CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                         DimsVectorUtils::Count(shape) * sizeof(float), nullptr, &ret);
    if (ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(ret)
        return Status(TNNERR_OPENCL_MEMALLOC_ERROR, "OpenCL malloc memory failed");
    }
    state_buffer->SetData(&cl_buffer);
    ret = ocl_context_->CommandQueue()->enqueueWriteBuffer(cl_buffer, CL_TRUE, 0,
            DimsVectorUtils::Count(shape) * sizeof(float), zero_buffer_data.data());
    if (ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(ret)
        return Status(TNNERR_OPENCL_API_ERROR, "OpenCL enqueueWriteBuffer failed");
    }

    // transfer from clBuffer to clImage
    ImageBufferConvertor convertor(opencl_runtime, ocl_context_->CommandQueue());
    std::shared_ptr<OpenCLMemory> state_memory;
    state_memory.reset(new OpenCLMemory(TNN_CL_IMAGE));
    state_memory->SetData(blob->GetHandle().base, false);
    DimsVector nhwc_buffer_shape = {sequence * batch, ALIGN_UP4(hidden_size) * 4 * num_directions, 1, 1};
    Status ret_convert = convertor.ConvertBufferToImage(
            state_buffer.get(), NHWC_BUFFER, nhwc_buffer_shape, state_memory.get(), true);
    CHECK_TNN_OK(ret_convert)

    return TNN_OK;
}

std::vector<DataFormat> OpenCLLSTMONNXLayerAcc::SupportDataFormat(DataType data_type, int dims_size, BlobType blob_type) {
    std::vector<DataFormat> support_list;
    if (dims_size >= 2) {
        support_list.push_back(DATA_FORMAT_CNH4);
    }
    return support_list;
}

REGISTER_OPENCL_ACC(LSTMONNX, LAYER_LSTMONNX);
REGISTER_OPENCL_LAYOUT(LAYER_LSTMONNX, DATA_FORMAT_CNH4);
}  // namespace TNN_NS
