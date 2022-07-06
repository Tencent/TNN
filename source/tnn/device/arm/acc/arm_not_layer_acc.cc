#include "tnn/device/arm/acc/arm_unary_layer_acc.h"

namespace TNN_NS {

typedef struct arm_not_operator : arm_unary_operator {
    virtual Float4 operator()(const Float4& v) {
        Float4 dst;
        dst.value[0] = (v.value[0] != 0 ? 0 : 1);
        dst.value[1] = (v.value[1] != 0 ? 0 : 1);
        dst.value[2] = (v.value[2] != 0 ? 0 : 1);
        dst.value[3] = (v.value[3] != 0 ? 0 : 1);
        return dst;
    }
} ARM_NOT_OP;

DECLARE_ARM_UNARY_ACC(Not, ARM_NOT_OP);

REGISTER_ARM_ACC(Not, LAYER_NOT);
REGISTER_ARM_LAYOUT(LAYER_NOT, DATA_FORMAT_NC4HW4);

}  // namespace TNN_NS
