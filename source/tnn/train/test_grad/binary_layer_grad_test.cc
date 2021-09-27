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

// author: sanerzheng@tencent.com

#include "tnn/train/test_grad/test_layer_grad.h"
#include "tnn/train/grad/utils.h"
#include "tnn/core/default_network.h"
#include "tnn/layer/multidir_broadcast_layer.h"
#include "tnn/utils/random_data_utils.h"


namespace TNN_NS {
namespace train {
DECLARE_LAYER_GRAD_TEST_BEGIN(Binary, LAYER_ADD); 
virtual Status TestAdd();
virtual Status TestSub();
virtual Status TestMul();
virtual Status TestDiv();
virtual Status TestBroadcast();
DECLARE_LAYER_GRAD_TEST_END

Status check_result(std::shared_ptr<RawBuffer> input_grad, std::shared_ptr<RawBuffer> expect_grad, float e = 2e-4) {
    auto expect_total = DimsVectorUtils::Count(expect_grad->GetBufferDims());
    auto input_total = DimsVectorUtils::Count(input_grad->GetBufferDims());
    if(expect_total != input_total) {
        LOGE("in and out dims's size not math");
        return Status(TNN_TRAIN_TEST_ERROR, "in and out dims's size not math");        
    }
    std::shared_ptr<RawBuffer> in_nchw;
    std::shared_ptr<RawBuffer> expect_nchw;
    ConvertToNCHW(input_grad, in_nchw);
    ConvertToNCHW(expect_grad, expect_nchw);
    float* input_ptr = in_nchw->force_to<float *>();
    float* expect_ptr = expect_nchw->force_to<float *>();
    for(int i=0; i<input_total; ++i) {
        if(std::abs(input_ptr[i] - expect_ptr[i]) > e) {
            LOGE("%d item error, input %f expect %f", i, input_ptr[i], expect_ptr[i]);
            return Status(TNN_TRAIN_TEST_ERROR, "binary test error");
        }
    }
    return TNN_OK;
}

Status test_binary_base(NameBuffers& inputs_buffer, NameBuffers& outputs_grad_buffer, std::vector<std::shared_ptr<RawBuffer>> expect_grad_buffer, RawBuffer& weight_buffer, DeviceType device_type,int weight_input_index, LayerType layer_type) {
    TrainContext context;
    context.network = new DefaultNetwork();
    NetworkConfig* config = new NetworkConfig();
    context.config = config;
    config->device_type = device_type;
    config->precision = PRECISION_HIGH;
    DataFormat data_format ;
    if(device_type == DEVICE_ARM)
        data_format = DATA_FORMAT_NC4HW4;
    else if(device_type == DEVICE_NAIVE)
        data_format = DATA_FORMAT_NCHW;
    else 
        return Status(TNN_TRAIN_TEST_ERROR, "not support device in test_concat_base");
    BaseLayer* cur_layer = new MultidirBroadcastLayer(layer_type); //fake for add/sub/mul/div layer
    cur_layer->layer_name_ = "test_binary";


    LayerParam* layer_param = new MultidirBroadcastLayerParam();
    dynamic_cast<MultidirBroadcastLayerParam*>(layer_param)->weight_input_index = weight_input_index;
    cur_layer->param_ = layer_param;

    LayerResource* layer_res = new EltwiseLayerResource();
    if(weight_buffer.GetDataCount() > 0) {
        dynamic_cast<EltwiseLayerResource *>(layer_res)->element_handle = weight_buffer;
        cur_layer->resource_ = layer_res;
    }
    Status status; 
    NameShapes buffer_shapes = {
        {outputs_grad_buffer[0].first, outputs_grad_buffer[0].second.GetBufferDims()}
    };
    generate_blob(cur_layer->input_blobs_, inputs_buffer, device_type, data_format, DATA_TYPE_FLOAT, true, true);
    generate_blob(cur_layer->output_blobs_, buffer_shapes, device_type, data_format, DATA_TYPE_FLOAT, false, false);
    cur_layer->InferOutputShape();
    outputs_grad_buffer[0].second.SetBufferDims(cur_layer->output_blobs_[0]->GetBlobDesc().dims);
    status = generate_raw_buffer(context.backward_grads_blob[cur_layer->output_blobs_[0]], outputs_grad_buffer[0].second, device_type, data_format, DATA_TYPE_FLOAT);
    RETURN_ON_NEQ(status, TNN_OK);
    
    status = LayerGrad::GetLayerGradMap()[layer_type]->OnGrad(cur_layer, context);
    RETURN_ON_NEQ(status, TNN_OK);
    auto output_grad0 = context.backward_grads_blob[cur_layer->output_blobs_[0]];
    output_buffer(output_grad0.get(), "binary output_grad");
    for(int i=0; i<cur_layer->input_blobs_.size(); ++i) {
        auto desc = cur_layer->input_blobs_[i]->GetBlobDesc();
        output_buffer(context.backward_grads_blob[cur_layer->input_blobs_[i]].get(), "binary " + desc.name + "_grad");
        LOGD("binary test %d %s", i, desc.name.c_str());
        if(weight_input_index == 0 && weight_buffer.GetDataCount() > 0)
            status = check_result(context.backward_grads_blob[cur_layer->input_blobs_[i]], expect_grad_buffer[1]);
        else
            status = check_result(context.backward_grads_blob[cur_layer->input_blobs_[i]], expect_grad_buffer[i]);
        RETURN_ON_NEQ(status, TNN_OK);
    }
    if(weight_buffer.GetDataCount() > 0){
        RawBuffer* weight_buffer = &(dynamic_cast<EltwiseLayerResource *>(layer_res)->element_handle);
        output_buffer(context.backward_grads_resource[weight_buffer].get(), "binary weight grad");
        LOGD("binary test weight grad");
        status = check_result(context.backward_grads_resource[weight_buffer], expect_grad_buffer[weight_input_index]);
        RETURN_ON_NEQ(status, TNN_OK);       
    }
    delete config;
    delete layer_param;
    delete layer_res;
    delete cur_layer;
    delete context.network;
    free_blobs(cur_layer->output_blobs_);
    free_blobs(cur_layer->input_blobs_);
    return TNN_OK; 
}

Status BinaryLayerGradTest::TestAdd() {
    RawBuffer input0(6 * sizeof(float), {2, 3});
    float input0_data[6] = {-0.0534f,  1.6219f,  0.3643f, -0.0289f, -1.4486f, -1.8167f};
    input0.buffer(reinterpret_cast<char *>(input0_data), 6 * sizeof(float));
    RawBuffer input1(2 * sizeof(float), {2,1});
    float input1_data[2] = {-1.3643, 1.0631};
    input1.buffer(reinterpret_cast<char *>(input1_data), 2 * sizeof(float));
    NameBuffers inputs_buffer = {{"input0", input0}, {"input1", input1}};

    RawBuffer output0_grad(6 * sizeof(float), {6,1});
    float output0_grad_data[6] = {0.0262, 0.0410, 0.0328, 0.0322, 0.0402, 0.0363};
    output0_grad.buffer(reinterpret_cast<char *>(output0_grad_data), 6 * sizeof(float));
    NameBuffers output0_grad_buffer = {{"output0", output0_grad}};

    std::shared_ptr<RawBuffer> expect0(new RawBuffer(6 * sizeof(float), {2, 3}));
    float expect0_data[6] = {0.0262, 0.0410, 0.0328, 0.0322, 0.0402, 0.0363};
    expect0->buffer(reinterpret_cast<char *>(expect0_data), 6 * sizeof(float));
    std::shared_ptr<RawBuffer> expect1(new RawBuffer(2* sizeof(float), {2,1}));
    float expect1_data[2] = {0.0999, 0.1087};
    expect1->buffer(reinterpret_cast<char *>(expect1_data), 2 * sizeof(float));
    std::vector<std::shared_ptr<RawBuffer>> expect_grad_buffers = {expect0, expect1};

    RawBuffer weight_buffer;
    LOGD("TestAdd 1 start ");
    
    Status status = test_binary_base(inputs_buffer, output0_grad_buffer, {expect0, expect1}, weight_buffer, DEVICE_ARM, 0, LAYER_ADD);
    RETURN_ON_NEQ(status, TNN_OK);
    LOGD("TestAdd 2 start ");
    status = test_binary_base(inputs_buffer, output0_grad_buffer, {expect0, expect1}, weight_buffer, DEVICE_NAIVE, 0, LAYER_ADD);
    
    weight_buffer = RawBuffer(2 * sizeof(float), {2,1});
    weight_buffer.buffer(reinterpret_cast<char *>(input1_data), 2 * sizeof(float));
    inputs_buffer = {{"input0", input0}};
    LOGD("TestAdd 3 start ");
    status = test_binary_base(inputs_buffer, output0_grad_buffer, {expect1, expect0}, weight_buffer, DEVICE_ARM, 0, LAYER_ADD);
    RETURN_ON_NEQ(status, TNN_OK);
    LOGD("TestAdd 4 start ");
    status = test_binary_base(inputs_buffer, output0_grad_buffer, {expect0, expect1}, weight_buffer, DEVICE_ARM, 1, LAYER_ADD);
    RETURN_ON_NEQ(status, TNN_OK);
    return TNN_OK;    
}

Status BinaryLayerGradTest::TestSub() {
    RawBuffer input0(6 * sizeof(float), {2, 3});
    float input0_data[6] = {-0.0534,  1.6219,  0.3643, -0.0289, -1.4486, -1.8167};
    input0.buffer(reinterpret_cast<char *>(input0_data), 6 * sizeof(float));
    RawBuffer input1(2 * sizeof(float), {2,1});
    float input1_data[2] = {-1.3643, 1.0631};
    input1.buffer(reinterpret_cast<char *>(input1_data), 2 * sizeof(float));
    NameBuffers inputs_buffer = {{"input0", input0}, {"input1", input1}};

    RawBuffer output0_grad(6 * sizeof(float), {6,1});
    float output0_grad_data[6] = {0.0262, 0.0410, 0.0328, 0.0322, 0.0402, 0.0363};
    output0_grad.buffer(reinterpret_cast<char *>(output0_grad_data), 6 * sizeof(float));
    NameBuffers output0_grad_buffer = {{"output0", output0_grad}};

    std::shared_ptr<RawBuffer> expect0(new RawBuffer(6 * sizeof(float), {2, 3}));
    float expect0_data[6] = {0.0262, 0.0410, 0.0328, 0.0322, 0.0402, 0.0363};
    expect0->buffer(reinterpret_cast<char *>(expect0_data), 6 * sizeof(float));
    std::shared_ptr<RawBuffer> expect1(new RawBuffer(2 * sizeof(float), {2,1}));
    float expect1_data[2] = {-0.0999, -0.1087};
    expect1->buffer(reinterpret_cast<char *>(expect1_data), 2 * sizeof(float));
    std::vector<std::shared_ptr<RawBuffer>> expect_grad_buffers = {expect0, expect1};

    RawBuffer weight_buffer;
    LOGD("TestSub 1 start ");
    
    Status status = test_binary_base(inputs_buffer, output0_grad_buffer, {expect0, expect1}, weight_buffer, DEVICE_ARM, 0, LAYER_SUB);
    RETURN_ON_NEQ(status, TNN_OK);
    LOGD("TestSub 2 start ");
    status = test_binary_base(inputs_buffer, output0_grad_buffer, {expect0, expect1}, weight_buffer, DEVICE_NAIVE, 0, LAYER_SUB);
    
    weight_buffer = RawBuffer(2 * sizeof(float), {2,1});
    weight_buffer.buffer(reinterpret_cast<char *>(input1_data), 2 * sizeof(float));
    inputs_buffer = {{"input0", input0}};
    LOGD("TestSub 3 start ");
    status = test_binary_base(inputs_buffer, output0_grad_buffer, {expect0, expect1}, weight_buffer, DEVICE_ARM, 1, LAYER_SUB);
    RETURN_ON_NEQ(status, TNN_OK);  
    return TNN_OK;
}

Status BinaryLayerGradTest::TestMul() {
    RawBuffer input0(6 * sizeof(float), {2, 3});
    float input0_data[6] = {-0.0534,  1.6219,  0.3643, -0.0289, -1.4486, -1.8167};
    input0.buffer(reinterpret_cast<char *>(input0_data), 6 * sizeof(float));
    RawBuffer input1(2 * sizeof(float), {2,1});
    float input1_data[2] = {-1.3643, 1.0631};
    input1.buffer(reinterpret_cast<char *>(input1_data), 2 * sizeof(float));
    NameBuffers inputs_buffer = {{"input0", input0}, {"input1", input1}};

    RawBuffer output0_grad(6 * sizeof(float), {6,1});
    float output0_grad_data[6] = {0.0416, 0.0148, 0.0392, 0.0417, 0.0242, 0.0184};
    output0_grad.buffer(reinterpret_cast<char *>(output0_grad_data), 6 * sizeof(float));
    NameBuffers output0_grad_buffer = {{"output0", output0_grad}};

    std::shared_ptr<RawBuffer> expect0(new RawBuffer(6 * sizeof(float), {2, 3}));
    float expect0_data[6] = {-0.0568, -0.0202, -0.0535, 0.0443,  0.0258,  0.0196};
    expect0->buffer(reinterpret_cast<char *>(expect0_data), 6 * sizeof(float));
    std::shared_ptr<RawBuffer> expect1(new RawBuffer(2 * sizeof(float), {2,1}));
    float expect1_data[2] = {0.0361, -0.0698};
    expect1->buffer(reinterpret_cast<char *>(expect1_data), 2 * sizeof(float));
    std::vector<std::shared_ptr<RawBuffer>> expect_grad_buffers = {expect0, expect1};

    RawBuffer weight_buffer;
    LOGD("TestMul 1 start ");
    
    Status status = test_binary_base(inputs_buffer, output0_grad_buffer, {expect0, expect1}, weight_buffer, DEVICE_ARM, 0, LAYER_MUL);
    RETURN_ON_NEQ(status, TNN_OK);
    LOGD("TestMul 2 start ");
    status = test_binary_base(inputs_buffer, output0_grad_buffer, {expect0, expect1}, weight_buffer, DEVICE_NAIVE, 0, LAYER_MUL);
    
    weight_buffer = RawBuffer(2 * sizeof(float), {2,1});
    weight_buffer.buffer(reinterpret_cast<char *>(input1_data), 2 * sizeof(float));
    inputs_buffer = {{"input0", input0}};
    LOGD("TestMul 3 start ");
    status = test_binary_base(inputs_buffer, output0_grad_buffer, {expect1, expect0}, weight_buffer, DEVICE_ARM, 0, LAYER_MUL);
    RETURN_ON_NEQ(status, TNN_OK);
    LOGD("TestMul 4 start ");
    status = test_binary_base(inputs_buffer, output0_grad_buffer, {expect0, expect1}, weight_buffer, DEVICE_ARM, 1, LAYER_MUL);
    RETURN_ON_NEQ(status, TNN_OK);
    return TNN_OK;
}

Status BinaryLayerGradTest::TestDiv() {
    RawBuffer input0(6 * sizeof(float), {2, 3});
    float input0_data[6] = {-0.0534,  1.6219,  0.3643, -0.0289, -1.4486, -1.8167};
    input0.buffer(reinterpret_cast<char *>(input0_data), 6 * sizeof(float));
    RawBuffer input1(2 * sizeof(float), {2,1});
    float input1_data[2] = {-1.3643, 1.0631};
    input1.buffer(reinterpret_cast<char *>(input1_data), 2 * sizeof(float));
    NameBuffers inputs_buffer = {{"input0", input0}, {"input1", input1}};

    RawBuffer output0_grad(6 * sizeof(float), {6,1});
    float output0_grad_data[6] = {0.0417, 0.0298, 0.0409, 0.0417, 0.0270, 0.0216};
    output0_grad.buffer(reinterpret_cast<char *>(output0_grad_data), 6 * sizeof(float));
    NameBuffers output0_grad_buffer = {{"output0", output0_grad}};

    std::shared_ptr<RawBuffer> expect0(new RawBuffer(6 * sizeof(float), {2, 3}));
    float expect0_data[6] = {-0.0305, -0.0219, -0.0300, 0.0392,  0.0254,  0.0204};
    expect0->buffer(reinterpret_cast<char *>(expect0_data), 6 * sizeof(float));
    std::shared_ptr<RawBuffer> expect1(new RawBuffer(2 * sizeof(float), {2,1}));
    float expect1_data[2] = {-0.0328, 0.0705};
    expect1->buffer(reinterpret_cast<char *>(expect1_data), 2 * sizeof(float));
    std::vector<std::shared_ptr<RawBuffer>> expect_grad_buffers = {expect0, expect1};

    RawBuffer weight_buffer;
    LOGD("TestDiv 1 start ");
    
    Status status = test_binary_base(inputs_buffer, output0_grad_buffer, {expect0, expect1}, weight_buffer, DEVICE_ARM, 0, LAYER_DIV);
    RETURN_ON_NEQ(status, TNN_OK);
    LOGD("TestDiv 2 start ");
    status = test_binary_base(inputs_buffer, output0_grad_buffer, {expect0, expect1}, weight_buffer, DEVICE_NAIVE, 0, LAYER_DIV);
    
    weight_buffer = RawBuffer(2 * sizeof(float), {2,1});
    weight_buffer.buffer(reinterpret_cast<char *>(input1_data), 2 * sizeof(float));
    inputs_buffer = {{"input0", input0}};
    LOGD("TestDiv 3 start ");
    status = test_binary_base(inputs_buffer, output0_grad_buffer, {expect0, expect1}, weight_buffer, DEVICE_ARM, 1, LAYER_DIV);
    RETURN_ON_NEQ(status, TNN_OK);  
    return TNN_OK;
}

Status BinaryLayerGradTest::TestBroadcast() {
    RawBuffer input0(6 * sizeof(float), {2, 3});
    float input0_data[6] = {-0.0534,  1.6219,  0.3643, -0.0289, -1.4486, -1.8167};
    input0.buffer(reinterpret_cast<char *>(input0_data), 6 * sizeof(float));
    RawBuffer input1(3 * sizeof(float), {1,3});
    float input1_data[3] = {-1.3643,1.0631, 0.9836};
    input1.buffer(reinterpret_cast<char *>(input1_data), 3 * sizeof(float));
    NameBuffers inputs_buffer = {{"input0", input0}, {"input1", input1}};

    RawBuffer output0_grad(6 * sizeof(float), {6,1});
    float output0_grad_data[6] = {0.0416, 0.0214, 0.0404, 0.0417, 0.0242, 0.0205};
    output0_grad.buffer(reinterpret_cast<char *>(output0_grad_data), 6 * sizeof(float));
    NameBuffers output0_grad_buffer = {{"output0", output0_grad}};

    std::shared_ptr<RawBuffer> expect0(new RawBuffer(6 * sizeof(float), {2, 3}));
    float expect0_data[6] = {-0.0568,  0.0228,  0.0397, -0.0568,  0.0258,  0.0201};
    expect0->buffer(reinterpret_cast<char *>(expect0_data), 6 * sizeof(float));
    std::shared_ptr<RawBuffer> expect1(new RawBuffer(3 * sizeof(float), {1,3}));
    float expect1_data[3] = {-0.0034, -0.0004, -0.0225};
    expect1->buffer(reinterpret_cast<char *>(expect1_data), 3 * sizeof(float));

    RawBuffer weight_buffer;
    LOGD("TestBroadcast 1 start ");
    
    Status status = test_binary_base(inputs_buffer, output0_grad_buffer, {expect0, expect1}, weight_buffer, DEVICE_ARM, 0, LAYER_MUL);
    RETURN_ON_NEQ(status, TNN_OK);
    LOGD("TestBroadcast 2 start ");
    status = test_binary_base(inputs_buffer, output0_grad_buffer, {expect0, expect1}, weight_buffer, DEVICE_NAIVE, 0, LAYER_MUL);
    
    weight_buffer = RawBuffer(3 * sizeof(float), {1,3});
    weight_buffer.buffer(reinterpret_cast<char *>(input1_data), 3 * sizeof(float));
    inputs_buffer = {{"input0", input0}};
    LOGD("TestBroadcast 3 start ");
    status = test_binary_base(inputs_buffer, output0_grad_buffer, {expect1, expect0}, weight_buffer, DEVICE_ARM, 0, LAYER_MUL);
    RETURN_ON_NEQ(status, TNN_OK);
    LOGD("TestBroadcast 4 start ");
    status = test_binary_base(inputs_buffer, output0_grad_buffer, {expect0, expect1}, weight_buffer, DEVICE_ARM, 1, LAYER_MUL);
    RETURN_ON_NEQ(status, TNN_OK);  
    return TNN_OK;
}

Status BinaryLayerGradTest::TestGrad() {
    Status status;
    status = TestAdd();
    RETURN_ON_NEQ(status, TNN_OK);
    status = TestSub();
    RETURN_ON_NEQ(status, TNN_OK);
    status = TestMul();
    RETURN_ON_NEQ(status, TNN_OK);
    status = TestDiv();
    RETURN_ON_NEQ(status, TNN_OK);
    status = TestBroadcast();
    RETURN_ON_NEQ(status, TNN_OK);
    return TNN_OK;
}
REGISTER_LAYER_GRAD_TEST(Binary, LAYER_ADD);
}
}