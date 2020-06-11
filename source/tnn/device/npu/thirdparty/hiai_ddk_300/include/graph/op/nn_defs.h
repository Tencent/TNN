#ifndef _CCE_GRAPH_OP_GE_OP_NN_DEFS_H
#define _CCE_GRAPH_OP_GE_OP_NN_DEFS_H

#include "../operator_reg.h"

namespace ge {
/**
 * The activation ops provide different types of nonlinearities for use in neural networks.
 * These include smooth nonlinearities (sigmoid, tanh, and softplus), continuous but not everywhere differentiable functions (relu, relu6, and relu_x), and random regularization (dropout).
 * <Input>
 *      x : The input tensor.
 * <Output>
 *      y : The output tensor that has the same type as the input tensor.
 * <Attr>
 *      mode : activation mode,there are sigmoid, ReLU, tanh, clipped ReLU, ELU, LEAKY_RELU, Abs, relu1, softsign, softplus, hardsigmoid, threshold, selu, linear.
 *      coef : The coef parameter indicates the upper limit value in the chipped RELU,
 *             and the alpha value in the ELU. It is not used in the original RELU. Fill in the default value of 0.0.
 */
REG_OP(Activation)
.INPUT(x, TensorType({ DT_INT8, DT_INT32, DT_FLOAT, DT_BOOL, DT_INT64 }))
.OUTPUT(y, TensorType({ DT_INT8, DT_INT32, DT_FLOAT, DT_BOOL, DT_INT64 }))
.ATTR(mode, AttrValue::INT { 1 })
.ATTR(coef, AttrValue::FLOAT { 0 })
.ATTR(negative_slope, AttrValue::FLOAT { 0.0 })
.OP_END()

/**
 * The BatchNorm operator normalizes the input to have 0-mean and/or unit (1) variance across the batch (batch normalization).
 * <Input>
 *      x       : The input tensor,
 *      scale   : A tensor for scaling factor, to scale the normalized x.
 *      b       : A tensor for bias, to shift to the normalized x.
 *      mean    : A tensor for population mean. Used for inference only; must be empty for training.
 *      variance: A tensor for population variance. Used for inference only; must be empty for training.
 * <Output>
 *      y : The output tensor.
 * <Attr>
 *      momentum         : Factor used in computing the running mean and variance.e.g.,
 *                         running_mean = running_mean * momentum + mean * (1 - momentum).
 *      epsilon          : A small float number added to the variance of x.
 *      mode             : BatchNorm mode, 0: bnScale, bnBias tensor dims are 1xCxHxW, 1: bnScale, bnBias tensor dims are 1xCx1x1.
 *      use_global_stats : A bool value to indicate the operation is for training (default) or inference.
 */
REG_OP(BatchNorm)
.INPUT(x, TensorType ({ DT_FLOAT }))
.OPTIONAL_INPUT(scale, TensorType ({ DT_FLOAT }))
.OPTIONAL_INPUT(b, TensorType ({ DT_FLOAT }))
.INPUT(mean, TensorType ({ DT_FLOAT }))
.INPUT(variance, TensorType ({ DT_FLOAT }))
.OUTPUT(y, TensorType ({ DT_FLOAT }))
.ATTR(momentum, AttrValue::FLOAT { 0.9 })
.ATTR(epsilon, AttrValue::FLOAT { 1e-5f })
.ATTR(mode, AttrValue::INT { 1 })
.ATTR(use_global_stats, AttrValue::BOOL { true })
.OP_END()

/**
 * The convolution operator consumes an input tensor and a filter, and computes the output.
 * <Input>
 *      x : The input tensor
 *      w : must be Const-OP. 
 *      b : must be Const-OP. Optional 1D bias to be added to the convolution, has size of M.
 * <Output>
 *      y : The output tensor.
 * <Attr>
 *      mode       : convolution mode, there are math convolution, cross-correlation convolution, deconvolution, depthwise convolution.
 *      group      : number of groups input channels and output channels are divided into
 *      num_output : number of output tensor must be nonnegative.
 *      pad        : Padding for the beginning and ending along each axis.
 *      stride     : Stride along each axis.
 *      dilation   : dilation value along each axis of the filter.
 *      kernel     : The shape of the convolution kernel. The size of kernel >= 2 and the value should be {0, 0, ...}, {0, w, ...}, {h, 0, ...} or {h, w, ...}, which h, w is Weights's [n,c,h,w].
 *      pad_mode   : pad mode, 0:NOTSET, 6:SAME 5:VALID. defaul default value is 0:NOTSET
 */
REG_OP(Convolution)
.INPUT(x, TensorType ({ DT_FLOAT }))
.INPUT(w, TensorType({ DT_FLOAT, DT_INT8 }))
.OPTIONAL_INPUT(b, TensorType ({ DT_FLOAT }))
.OUTPUT(y, TensorType ({ DT_FLOAT }))
.ATTR(mode, AttrValue::INT { 1 })
.ATTR(group, AttrValue::INT { 1 })
.ATTR(num_output, AttrValue::INT { 0 })
.ATTR(pad, AttrValue::LIST_INT({ 0, 0,  0, 0 }))
.ATTR(stride, AttrValue::LIST_INT({ 1, 1 }))
.ATTR(dilation, AttrValue::LIST_INT({ 1, 1 }))
.ATTR(kernel, AttrValue::LIST_INT({ 0, 0 }))
.ATTR(pad_mode, AttrValue::INT { 0 })
.OP_END()

/**
 * The Eltwise operator add all input tensors element wise.
 * <Input>
 *      x1 : The input tensor, must all be the same size and shape.
 *      x2 : The input tensor, must all be the same size and shape.
 * <Output>
 *      y : The output tensor, has same element type as two inputs.
 * <Attr>
 *      mode  : 0:product,1:sum,2:max;default is CC_ELTWISE_SUM.
 *      coeff : input_num should be equal with coeff size.
 *      weight: Since the MaskRCNN network needs to use Eltwise instead of the ADD operator, it is necessary to consider the processing of weights.
 */
REG_OP(Eltwise)
.INPUT(x1, TensorType({ DT_FLOAT, DT_BOOL }))
.INPUT(x2, TensorType({ DT_FLOAT, DT_BOOL }))
.OUTPUT(y, TensorType({ DT_FLOAT, DT_BOOL }))
.ATTR(mode, AttrValue::INT { 1 })
.ATTR(coeff, AttrValue::LIST_FLOAT({ 1, 1 }))
.ATTR(weight, AttrValue::LIST_TENSOR {})
.OP_END()

/**
 * The LRN operator means Local Response Normalization.
 * <Input>
 *      x : the input tensor.
 *      w : Dynamic input tensor.
 * <Output>
 *      y : Output tensor.
 * <Attr>
 *      lrn_localsize: The number of channels to sum over.
 *      lrn_alpha    : A scale factor, usually positive.
 *      lrn_beta     : An exponent.
 *      lrn_k        : An offset (usually positive to avoid dividing by 0).
 */
REG_OP(LRN)
.INPUT(x, TensorType({ DT_FLOAT }))
.DYNAMIC_INPUT(w, TensorType({ DT_FLOAT }))
.OUTPUT(y, TensorType({ DT_FLOAT }))
.ATTR(lrn_localsize, AttrValue::INT { 5 })
.ATTR(lrn_alpha, AttrValue::FLOAT { 1.0 })
.ATTR(lrn_beta, AttrValue::FLOAT { 0.5 })
.ATTR(lrn_k, AttrValue::FLOAT { 1.0 })
.OP_END()

/**
 * The ConvolutionDepthwise operator Computes a depthwise convolution from given input and filter tensors.
 * <Input>
 *      x     : the input tensor.
 *      filter: must be Const-OP.
 * <Output>
 *      y : the output has in_channels * channel_multiplier channels.
 * <Attr>
 *      num_output    : number of output tensor must be nonnegative.
 *      group         : number of groups input channels and output channels are divided into.
 *      pad_mode      : must be 5:CC_PADDING_VALID_NEW or 6: CC_PADDING_SAME_NEW
 *      mode          : ConvolutionDepthwise mode: matrix gemm algo, winograd Transform algo.
 *      algo          : algo type: 0: matrix gemm algo, 1: Winograd Transform algo, 2: accumulate in L0c with FP32.
 *      pad           : Padding for the beginning and ending along each axis.
 *      stride        : The stride of the sliding window for each dimension of input.
 *      dilation      : dilation value along each spatial axis of the filter.
 *      kernel        : The shape of the convolution kernel.
 *      format        : Specify the data format of the input and output data.
 *                      With the default format "NHWC", the data is stored in the order of: [batch, height, width, channels].
 *                      Alternatively, the format could be "NCHW", the data storage order of: [batch, channels, height, width].
 */
REG_OP(ConvolutionDepthwise)
.INPUT(x, TensorType({ DT_FLOAT }))
.INPUT(filter, TensorType({ DT_FLOAT }))
.OUTPUT(y, TensorType({ DT_FLOAT }))
.ATTR(num_output, AttrValue::INT{0})
.ATTR(group, AttrValue::INT{1})
.ATTR(pad_mode, AttrValue::INT{6})
.ATTR(mode, AttrValue::INT{1})
.ATTR(algo, AttrValue::INT{0})
.ATTR(pad, AttrValue::LIST_INT({ 0, 0, 0, 0}))
.ATTR(stride, AttrValue::LIST_INT({ 1, 1 }))
.ATTR(dilation, AttrValue::LIST_INT({ 1, 1 }))
.ATTR(kernel, AttrValue::LIST_INT{})
.ATTR(format, AttrValue::INT{0})
.OP_END()

/**
 * The FullConnection operator computes an inner product with a input tensor, a set of learned weights and adds biases.
 * <Input>
 *      x : the input tensor
 *      w : the weight tensor.
 *      b : A tensor for bias.
 * <Output>
 *      y : the output tensor
 * <Attr>
 *      num_output : The number of neurons output after full connection.
 */
REG_OP(FullConnection)
.INPUT(x, TensorType({ DT_FLOAT }))
.INPUT(w, TensorType({ DT_FLOAT }))
.OPTIONAL_INPUT(b, TensorType({ DT_FLOAT, DT_INT32}))
.OUTPUT(y, TensorType({ DT_FLOAT }))
.ATTR(num_output, AttrValue::INT { 0 })
.OP_END()

/**
 * Pools the input tensors by taking the max, average, etc. within regions.
 * <Input>
 *      x : The input tensor,
 * <Output>
 *      y : The output tensor.
 * <Attr>
 *      mode           : 0:max pooling   1:avg pooling  2:L2 pooling
 *      pad_mode       : pad mode, 0:NOTSET, 6:SAME 5:VALID. defaul default value is 0:NOTSET
 *      global_pooling : tensorflow have no attr, set default value
 *      window         : window size, specifies height and width of each filter. Here the size must be 2 and value >= 1.
 *      pad            : pad size, specifies the number of pixels to (implicitly) add to each side of the input. Here the size must be 4 and value >= 0.
 *      stride         : stride size, specifies the intervals at which to apply the filters to the input.Here the size must be 2 and value >= 1.
 *      ceil_mode      : pooling ceil mode, 0: DOMI_POOLING_CEIL, 1:DOMI_POOLING_FLOOR
 *      data_mode      : data_mode, DOMI_CAFFE_DATA_MODE =0, TENSORFLOW_DATA_MODE = 1.
 */
REG_OP(Pooling)
.INPUT(x, TensorType({ DT_FLOAT }))
.OUTPUT(y, TensorType({ DT_FLOAT }))
.ATTR(mode, AttrValue::INT { 0 })
.ATTR(pad_mode, AttrValue::INT { 0 })
.ATTR(global_pooling, AttrValue::BOOL { false })
.ATTR(window, AttrValue::LIST_INT({ 1, 1 }))
.ATTR(pad, AttrValue::LIST_INT({ 0, 0,  0,0 }))
.ATTR(stride, AttrValue::LIST_INT({ 1, 1 }))
.ATTR(ceil_mode, AttrValue::INT { 0 })
.ATTR(data_mode, AttrValue::INT { 1 })
.OP_END()

/**
 * Computes the elementwise product of input tensors.
 * <Input>
 *      x      : The input tensor,
 *      filter : must come from Const-OP.
 *      bias   : bias must come from Const-OP.
 * <Output>
 *      y : The output tensor, have the same shape as x, then computing the elementwise product.

 * <Attr>
 *      axis           : [default = 1] The first axis of input tensor along which to apply other tensors(filter or bias).
 *                       May be negative to index from the end (e.g., -1 for the last axis).
 *      bias_term      : [default = false] Whether to also learn a bias.
 *      has_bias_value : [default = false].
 *      filler_type    : Type of constant. Scale has no weight, does not call ParserWeight, filler training, you need to manually fill.
 *      filler_value   : filler is ignored unless just one bottom is given and the scale is a learned parameter of the layer.
 */
REG_OP(Scale)
.INPUT(x, TensorType ({ DT_FLOAT }))
.INPUT(filter, TensorType ({ DT_FLOAT }))
.OPTIONAL_INPUT(bias, TensorType ({ DT_FLOAT }))
.OUTPUT(y, TensorType ({ DT_FLOAT }))
.ATTR(axis, AttrValue::INT { 1 })
.ATTR(bias_term, AttrValue::BOOL { false })
.ATTR(has_bias_value, AttrValue::BOOL { false })
.ATTR(filler_type, AttrValue::STR { "constant" })
.ATTR(filler_value, AttrValue::FLOAT { 1.0 })
.OP_END()

/**
 * The ShuffleChannel with two stacked group convolutionstons.
 * <Input>
 *      x : The input tensor,
 * <Output>
 *      y : Each output channel only relates to the input channels within the group.
 * <Attr>
 *      group : Number of groups that input channels and output channels are divided into, it must be positive numbers.
 */
REG_OP(ShuffleChannel)
.INPUT(x, TensorType({ DT_FLOAT, DT_INT8 }))
.OUTPUT(y, TensorType({ DT_FLOAT, DT_INT8 }))
.ATTR(group, AttrValue::INT { 1 })
.OP_END()

/**
 * The operator computes the softmax (normalized exponential) values for each layer in the batch of the given input.
 * <Input>
 *      x : The input tensor,
 * <Output>
 *      y : The output values with the same shape as input tensor.
 * <Attr>
 *      axis : the dims to be used in softmax.
 *      algo : now is only support 1.
 *             1 means using "subtract max from every point to avoid overflow",
 *             0 means using "ubtract max from every point to avoid overflow",
 *             2 means using "perform the Log softmax operation to avoid overflow"
 */
REG_OP(Softmax)
.INPUT(x, TensorType ({ DT_FLOAT }))
.OUTPUT(y, TensorType ({ DT_FLOAT }))
.ATTR(axis, AttrValue::INT { 0 })
.ATTR(algo, AttrValue::INT { 1 })
.OP_END()

}  // namespace ge

#endif  // _CCE_GRAPH_OP_GE_OP_NN_DEFS_H
