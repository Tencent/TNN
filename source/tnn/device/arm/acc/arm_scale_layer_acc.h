//  Copyright © 2019 tencent. All rights reserved.

#ifndef TNN_SOURCE_TNN_DEVICE_ARM_ARM_SCALE_LAYER_ACC_H_
#define TNN_SOURCE_TNN_DEVICE_ARM_ARM_SCALE_LAYER_ACC_H_

#include "tnn/device/arm/acc/arm_batch_norm_layer_acc.h"

namespace TNN_NS {

// @brief conv layer cpu acc
class ArmScaleLayerAcc : public ArmBatchNormLayerAcc {
public:
    virtual ~ArmScaleLayerAcc();

    virtual Status allocateBufferParam(const std::vector<Blob *> &inputs,
                                       const std::vector<Blob *> &outputs);
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_ARM_ARM_SCALE_LAYER_ACC_H_
