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

#ifndef TNN_SOURCE_TNN_DEVICE_ARM_ARM_LAYER_ACC_H_
#define TNN_SOURCE_TNN_DEVICE_ARM_ARM_LAYER_ACC_H_

#include <string>
#include <vector>

#include "tnn/core/abstract_layer_acc.h"
#include "tnn/core/macro.h"
#include "tnn/device/arm/arm_common.h"
#include "tnn/device/arm/arm_context.h"
#include "tnn/device/arm/arm_device.h"
#include "tnn/device/arm/arm_util.h"

namespace TNN_NS {
// @brief conv layer arm acc
class ArmLayerAcc : public AbstractLayerAcc {
public:
    virtual Status Init(Context *context, LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                const std::vector<Blob *> &outputs);

    virtual ~ArmLayerAcc();

    /**
     * @brief input or output blobs reshape.
     * @param inputs    input blobs
     * @param outputs   output blobs
     * @return reshape result
     */
    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

    /**
     * @brief layer forward
     * @param inputs    input blobs
     * @param outputs   output blobs
     * @return forward result
     */
    virtual Status Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

    /**
     * @brief layer Doforward
     * @param inputs    input blobs
     * @param outputs   output blobs
     * @return execution result
     */
    virtual Status DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

#if TNN_PROFILE
    Timer timer;
#endif

protected:
    LayerParam *param_       = nullptr;
    LayerResource *resource_ = nullptr;

    ArmContext *context_                     = nullptr;
    std::shared_ptr<ArmKernelParam> k_param_ = nullptr;

    virtual bool DataTypeSupported(DataType data_type);

private:
    // @brief return device layer acc support data format
    virtual std::vector<DataFormat> SupportDataFormat(DataType data_type, int dims_size);
};

#if TNN_ARM82
#define DECLARE_ARM_FP16_LAYER_FUNC                                                                                    \
    Status ExecFp16(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
#else
#define DECLARE_ARM_FP16_LAYER_FUNC
#endif  // TNN_ARM82

#define DECLARE_ARM_ACC(type_string, layer_type)                                                                       \
    class Arm##type_string##LayerAcc : public ArmLayerAcc {                                                            \
    public:                                                                                                            \
        virtual ~Arm##type_string##LayerAcc(){};                                                                       \
        virtual Status DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);               \
                                                                                                                       \
    private:                                                                                                           \
        template <typename T>                                                                                          \
        Status Exec(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);                            \
        DECLARE_ARM_FP16_LAYER_FUNC;                                                                                   \
    }

#define REGISTER_ARM_ACC(type_string, layer_type)                                                                      \
    ArmTypeLayerAccRegister<TypeLayerAccCreator<Arm##type_string##LayerAcc>> g_arm_##layer_type##_acc_register(        \
        layer_type);

class ArmTypeLayerFp16PrecisionCreator {
public:
    static std::shared_ptr<ImplementedPrecision> UpdateImplementedPrecision(LayerType layer_type) {
        // make sure arm device has been registered
        TypeDeviceRegister<ArmDevice> arm_device_register(DEVICE_ARM);
        auto implemented_precision          = GetDevice(DEVICE_ARM)->GetImplementedPrecision(layer_type);
        auto updated_precision              = std::make_shared<ImplementedPrecision>(*implemented_precision);
        updated_precision->fp16_implemented = true;
        return updated_precision;
    };
};

#if TNN_ARM82
#define REGISTER_ARM_PRECISION_FP16(layer_type)                                                                        \
    ArmTypeLayerPrecisionRegister g_arm_##layer_type##_fp16_precision_register(                                        \
        layer_type, ArmTypeLayerFp16PrecisionCreator::UpdateImplementedPrecision(layer_type));
#else
#define REGISTER_ARM_PRECISION_FP16(layer_type)
#endif  // TNN_ARM82

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_ARM_ARM_LAYER_ACC_H_
