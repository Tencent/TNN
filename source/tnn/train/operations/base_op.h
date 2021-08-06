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

#ifndef TNN_SOURCE_TNN_TRAIN_OPERATIONS_BASE_OP_H
#define TNN_SOURCE_TNN_TRAIN_OPERATIONS_BASE_OP_H
#include <vector>
#include "tnn/train/grad/utils.h"
#include "tnn/train/operations/op_type.h"

namespace TNN_NS {
namespace train {
// @brief base op, could be grad ops or ops like add

class BaseOp {
public:
    typedef std::vector<std::pair<int, std::shared_ptr<BaseOp>>> PriorityOp;
    BaseOp() {};
    virtual ~BaseOp() = default;
    virtual bool IsSupported(ParamWrappers& inputs, ParamWrappers& outputs, TrainContext& context) = 0; 
    virtual Status Exec(ParamWrappers& inputs, ParamWrappers& outputs, ParamWrappers& params) = 0;
    inline static std::map<OpType, PriorityOp>& GetPriorityOpTypeMap() {
        static std::map<OpType, PriorityOp > priority_op_type_map;
        return priority_op_type_map;
    };
    // inline static std::map<LayerType, OpType>& GetLayer2OpMap() {
    //     static std::map<LayerType, OpType> layer_2_op_map;
    //     return layer_2_op_map;
    // };
    
    static Status RunOp(OpType type, ParamWrappers& inputs, ParamWrappers& outputs, ParamWrappers& params, TrainContext& context){
        std::shared_ptr<BaseOp> op = GetSupportedOp(type, inputs, outputs, context);
        if(op == nullptr) 
            return Status(TNN_OP_NOT_FOUND, "not supported op type");
        return op->Exec(inputs, outputs, params);
    };
    // these funcs are for right value parameter
    static Status RunOp(OpType type, ParamWrappers&& inputs, ParamWrappers&& outputs, ParamWrappers& params, TrainContext& context){
        return RunOp(type, inputs, outputs, params, context);
    };
    static Status RunOp(OpType type, ParamWrappers& inputs, ParamWrappers&& outputs, ParamWrappers&& params, TrainContext& context){
        return RunOp(type, inputs, outputs, params, context);
    };
    static Status RunOp(OpType type, ParamWrappers&& inputs, ParamWrappers& outputs, ParamWrappers&& params, TrainContext& context){
        return RunOp(type, inputs, outputs, params, context);
    };
    static Status RunOp(OpType type, ParamWrappers&& inputs, ParamWrappers& outputs, ParamWrappers& params, TrainContext& context){
        return RunOp(type, inputs, outputs, params, context);
    };
    static Status RunOp(OpType type, ParamWrappers& inputs, ParamWrappers&& outputs, ParamWrappers& params, TrainContext& context){
        return RunOp(type, inputs, outputs, params, context);
    };
    static Status RunOp(OpType type, ParamWrappers& inputs, ParamWrappers& outputs, ParamWrappers&& params, TrainContext& context){
        return RunOp(type, inputs, outputs, params, context);
    };
    static Status RunOp(OpType type, ParamWrappers&& inputs, ParamWrappers&& outputs, ParamWrappers&& params, TrainContext& context){
        return RunOp(type, inputs, outputs, params, context);
    };

    static std::shared_ptr<BaseOp> GetSupportedOp(OpType type, ParamWrappers& inputs, ParamWrappers& outputs, TrainContext& context) {
        auto priority_op_type_map = GetPriorityOpTypeMap();
        auto iter = priority_op_type_map.find(type);
        if( iter == priority_op_type_map.end())
            return nullptr;
        for(auto priority_op_iter = iter->second.begin(); priority_op_iter != iter->second.end(); ++priority_op_iter) {
            if(priority_op_iter->second->IsSupported(inputs, outputs, context)) {
                return priority_op_iter->second;
            }
        }
        return nullptr;     
    };
    // static std::shared_ptr<BaseOp> GetOpByLayerType(LayerType type) {
    //     auto layer_2_op_map = GetLayer2OpMap();
    //     if(layer_2_op_map.find(type) == layer_2_op_map.end())
    //         return nullptr;
    //     return GetOpByOpType(layer_2_op_map[type]);
    // };
    static void RegisterOp(OpType type, int priority, std::shared_ptr<BaseOp> op){
        auto priority_op_type_map = BaseOp::GetPriorityOpTypeMap();
        auto iter = priority_op_type_map.find(type);
        std::pair<int, std::shared_ptr<BaseOp> > tmp_priority_op = std::make_pair(priority, op);
        if(iter == priority_op_type_map.end()) {
            priority_op_type_map[type] = {std::move(tmp_priority_op)};
        } else {
            auto priority_op_iter = iter->second.begin();
            for(; priority_op_iter != iter->second.end(); ++priority_op_iter) {
                if(priority_op_iter->first < priority) break;
            }
            iter->second.insert(priority_op_iter, std::move(tmp_priority_op));
        }
    };
};
#define DECLARE_OP(type_string, op_type)                                                                       \
class type_string##Op : public BaseOp {                                                            \
public:                                                                                                            \
    virtual ~##type_string##Op(){}; \
    virtual bool IsSupported(ParamWrappers& inputs, ParamWrappers& outputs, TrainContext& context);   \
    virtual Status Exec(ParamWrappers& inputs, ParamWrappers& ouputs, ParamWrappers& other_params);    \
}

template <typename T>
class OpRegister {
public:
    explicit OpRegister(OpType type, int priority) {
        BaseOp::RegisterOp(type, priority, std::make_shared<T>());
    }
};

#define REGISTER_OP(type_string, op_type, priority)                                                                      \
    OpRegister<##type_string##Op> g_##type_string##_grad_op_register(op_type, priority);

// template <typename T>
// class GradOpRegister {
// public:
//     explicit GradOpRegister(OpType type, LayerType layer_type, int priority) {
//         BaseOp::RegisterOp(type, priority, std::make_shared<T>());
//         BaseOp::GetLayer2OpMap[layer_type] =
//     }
// };

// #define REGISTER_GRAD_OP(type_string, op_type, layer_type, priority)                                                                      \
//     GradOpRegister<##type_string##Op> g_##type_string##_grad_op_register(layer_type, op_type, priority);



} // namespace train
} // namespace TNN_NS
#endif  //TNN_SOURCE_TNN_TRAIN_OPERATIONS_OP_TYPE_H 