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

#include "tnn/device/arm/acc/gradient/arm_binary_grad.h"
#include "tnn/device/arm/acc/arm_binary_layer_acc.h"
#include "tnn/layer/multidir_broadcast_layer.h"

namespace TNN_NS {

// z = x * y
// dz/dx = y
// dz/dy = x
typedef struct arm_mul_grad_function: arm_binary_grad_function {
    virtual std::pair<float, float> operator()(const float &i_0, const float &i_1, const float &o, const float &og) {
        return {og * i_1, og * i_0};
    }
    virtual std::pair<Float4, Float4> operator()(const Float4 &i_0, const Float4 &i_1, const Float4 &o,
                                                 const Float4 &og) {
        return {og * i_1, og * i_0};
    }
} ARM_MUL_GRAD_FUNC;

// DEFINE_ARM_BINARY_GRAD_OP(Mul, ARM_MUL_GRAD_FUNC)

class ArmMulGradOp : public GradOp {                                                                   
public:                                                                                                            
    virtual ~ArmMulGradOp(){};                                                                         
    virtual Status OnGrad(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs,                   
                          LayerResource *resource, GradientParam *grad_param, Context *context,                            
                          const GradOpInfo &grad_info) {                                                           
        auto forward = dynamic_cast<MultidirBroadcastLayer *>(grad_param->forward_layer);
        CHECK_PARAM_NULL(forward);
        
        auto forward_acc = dynamic_cast<ArmBinaryLayerAcc *>(forward->GetAbstractLayerAcc());
        CHECK_PARAM_NULL(forward_acc);
        
        auto forward_param = dynamic_cast<MultidirBroadcastLayerParam *>(grad_param->forward_param);
        CHECK_PARAM_NULL(forward_param);

        std::vector<Blob *> &fw_inputs = forward->GetInputBlobs();  // x
        int I = fw_inputs.size();
        std::vector<Blob *> input_grads(outputs.begin(), outputs.begin() + I);   // dL/dx 求导目标

        std::vector<Blob *> &fw_outputs = forward->GetOutputBlobs();   // y
        int O = fw_outputs.size();
        std::vector<Blob *> output_grads(inputs.begin() + I + O, inputs.begin() + I + O * 2);  // dL/dy

        int R = grad_param->need_train ? resource->GetTrainable().size() : 0;   // w
        std::vector<RawBuffer *> fw_resources;
        if (R > 0) {
            fw_resources = resource->GetTrainable();
        };                                                
        std::vector<Blob *> resource_grads(outputs.begin() + I, outputs.begin() + I + R);   // dL/dw 求导目标

        // 求导目标  dL/dx + dL/dw
        std::vector<void *> grad_ptrs;
        std::vector<bool> accumulate_grad;
        if (forward_acc->GetResource().GetDataCount() > 0) {
            if (forward_param->weight_input_index == 0) { // 参数作为左值
                grad_ptrs.push_back(GetBlobHandlePtr(resource_grads[0]->GetHandle()));
                grad_ptrs.push_back(GetBlobHandlePtr(input_grads[0]->GetHandle()));

                accumulate_grad.push_back(grad_info.accumulate_resource_grad[0]);
                accumulate_grad.push_back(grad_info.accumulate_input_grad[0]);
            } else {                                      // 参数作为右值
                grad_ptrs.push_back(GetBlobHandlePtr(input_grads[0]->GetHandle()));
                grad_ptrs.push_back(GetBlobHandlePtr(resource_grads[0]->GetHandle()));

                accumulate_grad.push_back(grad_info.accumulate_input_grad[0]);
                accumulate_grad.push_back(grad_info.accumulate_resource_grad[0]);
            }
        } else {
            grad_ptrs.push_back(GetBlobHandlePtr(input_grads[0]->GetHandle()));
            grad_ptrs.push_back(GetBlobHandlePtr(input_grads[1]->GetHandle()));
            accumulate_grad = grad_info.accumulate_input_grad;
        }

        // 前向的输入：x+w
        std::vector<void *> &forward_input_ptrs = forward_acc->GetInputPtrs();
        std::vector<DimsVector> &forward_input_dims = forward_acc->GetInputShapes();

        // 目前只处理float
        if (fw_inputs[0]->GetBlobDesc().data_type != DATA_TYPE_FLOAT) {
            LOGE("Arm Mul GradOp::OnGrad, dtype not supported\n");                                      
            return Status(TNNERR_TRAIN_ERROR, "dtype not supported");
        }

        int count_quad = DimsFunctionUtils::GetNCHWXPackedCount(fw_outputs[0]->GetBlobDesc().dims, 4);

        float* x0 = reinterpret_cast<float *>(forward_input_ptrs[0]);
        float* x1 = reinterpret_cast<float *>(forward_input_ptrs[1]);
        float* y = reinterpret_cast<float *>(GetBlobHandlePtr(fw_outputs[0]->GetHandle()));
        float* y_grad  = reinterpret_cast<float *>(GetBlobHandlePtr(output_grads[0]->GetHandle()));
        float* x0_grad = reinterpret_cast<float *>(grad_ptrs[0]);
        float* x1_grad = reinterpret_cast<float *>(grad_ptrs[0]);
        bool acc_x0 = accumulate_grad[0];
        bool acc_x1 = accumulate_grad[1];
        if (forward_acc->GetBroadCastType() == BroadcastTypeNormal) {
            ExecGradNormally(count_quad, x0_grad, x1_grad, x0, x1, y, y_grad, accumulate_grad[0], accumulate_grad[1]);
        } else if (forward_acc->GetBroadCastType() != BroadcastTypeSingle) {
            if (DimsVectorUtils::Count(forward_input_dims[0]) == 1) {  // x0是单个元素
                OMP_PARALLEL_FOR_
                for (int n = 0; n < count_quad; n++) {
                    auto in_0 = Float4(x0[0]);
                    auto in_1 = Float4::load(x1 + n * 4);
                    auto out  = Float4::load(y + n * 4); 
                    auto out_grad = Float4::load(y_grad + n * 4);  
                    auto in_grads = ARM_MUL_GRAD_FUNC()(in_0, in_1, out, out_grad);
                    auto& in_grad_0     = in_grads.first;                                                                            
                    auto& in_grad_1     = in_grads.second;    
                    // tmp_grad这个里面去实现累加                                                                      
                    // Float4::save(x0_grad + n * 4, acc_x0 ? (in_grad_0 + Float4::load(x0_grad + n * 4)) : in_grad_0);  
                    // Float4::save(x1_grad + n * 4, acc_x1 ? (in_grad_1 + Float4::load(x1_grad + n * 4)) : in_grad_1); 
                    // x0_grad += tmp_grad;
                    // VEC::save(output_ptr + n * pack, binary_op<op_type, VEC>(v2, v1, alpha, beta));
                }
            } else {
                OMP_PARALLEL_FOR_
                for (int n = 0; n < count_quad; n++) {
                    auto in_0 = Float4::load(x0 + n * 4);
                    auto in_1 = Float4(x1[0]);
                    auto out = Float4::load(y + n * 4); 
                    auto out_grad = Float4::load(y_grad + n * 4);  
                    auto in_grads = ARM_MUL_GRAD_FUNC()(in_0, in_1, out, out_grad);
                    auto& in_grad_0     = in_grads.first;                                                                            
                    auto& in_grad_1     = in_grads.second;                                                                           
                    // Float4::save(x0_grad + n * 4, acc_x0 ? (in_grad_0 + Float4::load(x0_grad + n * 4)) : in_grad_0);  
                    // Float4::save(x1_grad + n * 4, acc_x1 ? (in_grad_1 + Float4::load(x1_grad + n * 4)) : in_grad_1); 
                }
            }
        } else  {
            LOGE("Arm Mul GradOp::OnGrad, broadcast type not supported");               
            return Status(TNNERR_TRAIN_ERROR, "broadcast type not supported");  
        }

        return TNN_OK;                                 
    }                                                                                                              

private:                                                                                                           
    void ExecGradNormally(int count_quad, float *x0_grad, float *x1_grad, float *x0, float *x1,         
                          float *y, float *y_grad, bool acc_x0, bool acc_x1) {                                                                 
        Float4 in_0, in_1, in_grad_0, in_grad_1, out, out_grad;                                                        
        OMP_PARALLEL_FOR_                                                                                              
        for (int n = 0; n < count_quad; ++n) {                                                                         
            in_0          = Float4::load(x0 + n * 4);                                                           
            in_1          = Float4::load(x1 + n * 4);                                                             
            out           = Float4::load(y + n * 4);                                                              
            out_grad      = Float4::load(y_grad + n * 4);        
            auto in_grads = ARM_MUL_GRAD_FUNC()(in_0, in_1, out, out_grad);
            in_grad_0     = in_grads.first;                                                                            
            in_grad_1     = in_grads.second;                                                                           
            Float4::save(x0_grad + n * 4, acc_x0 ? (in_grad_0 + Float4::load(x0_grad + n * 4)) : in_grad_0);  
            Float4::save(x1_grad + n * 4, acc_x1 ? (in_grad_1 + Float4::load(x1_grad + n * 4)) : in_grad_1); 
        }                                                                                                        
    }
};

REGISTER_ARM_GRAD_OP(Mul, LAYER_MUL)
REGISTER_ARM_GRAD_LAYOUT(LAYER_MUL, DATA_FORMAT_NC4HW4)

}  // namespace TNN_NS
