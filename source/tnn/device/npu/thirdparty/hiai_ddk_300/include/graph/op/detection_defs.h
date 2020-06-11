#ifndef _CCE_GRAPH_OP_GE_OP_DETECTION_DEFS_H
#define _CCE_GRAPH_OP_GE_OP_DETECTION_DEFS_H

#include "../operator_reg.h"

namespace ge {
/**
 * Permutes the dimensions of the input according to a given pattern.
 * <Input>
 *      x : The input tensor, with the type of Data.
 *      w : The input tensor, with the type of Const.
 * <Output>
 *      y : Same as the input shape, but with the dimensions re-ordered according to the specified pattern.
 * <Attr>
 *      order : Permutation pattern. The size >= dims of x, a tuple of dimension indices, e.g. x:(4, 4, 4),thus order: (0, 2, 1).
 */
REG_OP(Permute)
.INPUT(x, TensorType ({ DT_FLOAT }))
.INPUT(w, TensorType ({ DT_FLOAT }))
.OUTPUT(y, TensorType ({ DT_FLOAT }))
.ATTR(order, AttrValue::LIST_INT { 0 })
.OP_END()

}  // namespace ge

#endif  // _CCE_GRAPH_OP_GE_OP_DETECTION_DEFS_H
