#ifndef _CCE_GRAPH_OP_GE_OP_MATH_DEFS_H
#define _CCE_GRAPH_OP_GE_OP_MATH_DEFS_H

#include "../operator_reg.h"

namespace ge {
/**
 * Performs tensor addition.
 * <Input>
 *      x1 : First operand.
 *      x2 : Second operand.
 * <Output>
 *      y  : Result, has same element type as two inputs.
 */
REG_OP(Add)
.INPUT(x1, TensorType({ DT_FLOAT }))
.INPUT(x2, TensorType({ DT_FLOAT })) 
.OUTPUT(y, TensorType({ DT_FLOAT }))
.OP_END()

/**
 * Performs tensor multiplication.
 * <Input>
 *      x : First operand.
 *      y : Second operand.
 * <Output>
 *      z : Result, has same element type as two inputs.
 */
REG_OP(Mul)
.INPUT(x, TensorType({ DT_FLOAT }))
.INPUT(y, TensorType({ DT_FLOAT }))
.OUTPUT(z, TensorType({DT_FLOAT}))
.OP_END()

}  // namespace ge

#endif  // _CCE_GRAPH_OP_GE_OP_MATH_DEFS_H
