#ifndef _CCE_GRAPH_OP_GE_OP_ARRAY_DEFS_H
#define _CCE_GRAPH_OP_GE_OP_ARRAY_DEFS_H

#include "../operator_reg.h"

namespace ge {

/**
 * A data tensor.
 * <Input>
 *      x : The input tensor, with the type of Data.
 * <Output>
 *      y : The output tensor.
 */
REG_OP(Data)
.INPUT(x, TensorType({DT_FLOAT, DT_INT8, DT_INT32, DT_BOOL}))
.OUTPUT(y, TensorType({DT_FLOAT, DT_INT8, DT_INT32, DT_BOOL}))
.OP_END()

/**
 * The function of Concat operator is concatenate a list of tensors into a single tensor along one dimension.
 * The number of dimensions of the input tensors must match, and all dimensions except axis must be equal.
 *  <Input>
 *      x    : List of tensors for concatenation.
 * <Output>
 *      y    : Concatenated tensor.
 * <Attr>
 *      axis : Which axis to concat on.
 *      N    : input nums
 */
REG_OP(Concat)
.DYNAMIC_INPUT(x, TensorType({ DT_FLOAT, DT_INT32 }))
.OUTPUT(y, TensorType({ DT_FLOAT, DT_INT32 }))
.ATTR(axis, AttrValue::INT { 1 })
.ATTR(N, AttrValue::INT { 1 })
.OP_END()

/**
 * The function of Reshape operator is reshape the input tensor.
 * <Input>
 *      tensor : The input tensor,with the type of Data.
 *      w      : Fill the shape with weight when the attribute shape is not entered,and the input type is const.
 * <Output>
 *      output : Reshaped tensor that has the same values as attr shape.
 * <Attr>
 *      shape    : The shape tensor,which specifies the output shape.
 *      axis     : Which axis to reshape.
 *      num_axes : Num_axes is used to calculate the output shape.
                   When num_axes is -1, output.shape.size() = shape.size() + axis.
                   When num_axes is another value, output.size() = shape.size() + tensor.size() - num_axes.
 */
REG_OP(Reshape)
.INPUT(tensor, TensorType({ DT_FLOAT, DT_INT32, DT_INT64, DT_BOOL }))
.OPTIONAL_INPUT(w, TensorType({ DT_FLOAT, DT_INT32, DT_INT64, DT_BOOL }))
.OUTPUT(output, TensorType({ DT_FLOAT, DT_INT32, DT_INT64, DT_BOOL }))
.REQUIRED_ATTR(shape, AttrValue::LIST_INT)
.ATTR(axis, AttrValue::INT { 0 })
.ATTR(num_axes, AttrValue::INT{ -1 })
.OP_END()

/**
 * The function of Split operator is split a tensor into a list of tensors along the axis.
 * <Input>
 *      x : The input tensor, with the type of Data.
 * <Output>
 *      y : Splited tensor. It is required and the value should equal to output_num.
 * <Attr>
 *      axis        : Which axis to split on. The value should be in range of [0, axis_dim_count) and axis_dim_count % output_num should be 0.
 *      output_num  : The count of tensor that you wish to split and axis_dim_count % output_num should be equal to 0.
 *      slice_point : Optional, used to specify the point of split,if set, slice_point.size = output_num - 1. 
 *                    The slice_point.value should be incremented, in range of (0, axis_dim_count).
 *                    Slice_point and the size_split should only set one of them or none of them. If both set, error will be reported.
 *      size_split  : Optional, used to specify the size of split, if set, SUM (size_split.value) = axis_dim_count. 
 */
REG_OP(Split)
.INPUT(x, TensorType({ DT_FLOAT, DT_INT8, DT_INT32, DT_BOOL }))
.DYNAMIC_OUTPUT(y, TensorType({ DT_FLOAT, DT_INT8, DT_INT32, DT_BOOL }))
.REQUIRED_ATTR(axis, AttrValue::INT)
.REQUIRED_ATTR(output_num, AttrValue::INT)
.ATTR(slice_point, AttrValue::LIST_INT {})
.ATTR(size_split, AttrValue::LIST_INT {})
.OP_END()

/**
 * Flattens the input tensor into a 2D matrix.
 * <Input>
 *      x : A tensor of rank.
 * <Output>
 *      y : A 2D tensor with the contents of the input tensor, with input dimensions up to axis flattened to the outer dimension of the output
 *          and remaining input dimensions flattened into the inner dimension of the output.
 */
REG_OP(Flatten)
.INPUT(x, TensorType({ DT_FLOAT }))
.OUTPUT(y, TensorType({ DT_FLOAT }))
.OP_END()

}  //namespace ge

#endif  // _CCE_GRAPH_OP_GE_OP_ARRAY_DEFS_H
