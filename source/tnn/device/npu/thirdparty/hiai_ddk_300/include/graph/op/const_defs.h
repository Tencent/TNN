#ifndef _CCE_GRAPH_OP_GE_OP_CONST_DEFS_H
#define _CCE_GRAPH_OP_GE_OP_CONST_DEFS_H

#include "../operator_reg.h"

namespace ge {
/**
 * A constant tensor.
 * <Output>
 *      y : Output tensor containing the same value of the provided tensor.
 * <Attr>
 *      value : The value for the elements of the output tensor.
 */
REG_OP(Const)
.OUTPUT(y, TensorType({DT_FLOAT, DT_INT8, DT_INT32, DT_BOOL}))
.ATTR(value, AttrValue::TENSOR(new (std::nothrow) Tensor(TensorDesc())))
.OP_END()

}  // namespace ge

#endif  // _CCE_GRAPH_OP_GE_OP_CONST_DEFS_H
