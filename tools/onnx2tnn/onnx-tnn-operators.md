# onnx vs tnn

以下表格中的内容,会随着后续工具的不断完善进行相应的补充

| onnx                                                         | tnn                        |
|--------------------------------------------------------------|---------------------------------|
| AveragePool                                                  | Pooling / Pooling3D             |
| BatchNormalization                                           | BatchNormCxx                    |
| Clip                                                         | ReLU6                           |
| Concat                                                       | Concat                          |
| Conv                                                         | Convolution3D / Convolution     |
| ConvTranspose(ConvTranspose+BatchNormalization)              | Deconvolution3D / Deconvolution |
| DepthToSpace                                                 | Reorg                           |
| Div                                                          | Mul                             |
| Gemm                                                         | InnerProduct                    |
| GlobalAveragePool                                            | Pooling / Pooling3D             |
| GlobalMaxPool                                                | Pooling / Pooling3D             |
| InstanceNormalization                                        | InstBatchNormCxx                |
| LeakyRelu                                                    | PReLU                           |
| MaxPool                                                      | Pooling / Pooling3D             |
| Mul                                                          | Mul                             |
| Normalize(ReduceL2 + Clip+ Expand+Div)                       | Normalize                       |
| PReLU                                                        | PReLU                           |
| Pad                                                          | Pad / Pad3D                     |
| ReduceMean                                                   | ReduceMean                      |
| Relu                                                         | ReLU                            |
| Reshape                                                      | Reshape                         |
| ShuffleChannle(Reshape+Transpose+Reshape)                    | ShuffleChannle                  |
| Slice                                                        | StridedSlice                    |
| Softmax(Exp + ReduceSum + Div)                               | SoftmaxCaffe                    |
| Softmax(Transpose + Reshape + Softmax + Reshape + Transpose) | SoftmaxCaffe                    |
| Softplus                                                     | Softplus                        |
| Split                                                        | SplitV                          |
| Sub                                                          | BatchNormCxx                    |
| Tanh                                                         | TanH                            |
| Tile                                                         | Repeat                          |
| Transpose                                                    | Transpose                       |
| Upsample                                                     | Upsample                        |

# use onnx operator

| onnx                  | 1.2.2                              | 1.6.0                                            | compatible   |
|-----------------------|------------------------------------|--------------------------------------------------|--------------|
| AveragePool           | -                                  | attributes(ceil\_mode)                           | yes          |
| BatchNormalization    | spatial                            | spatial(delete) (not use)                        | yes          |
| Clip                  | attributes(min, max)               | inputs(min, max)                                 | yes(support) |
| Concat                | -                                  | -                                                | yes          |
| Conv                  | -                                  | -                                                | yes          |
| ConvTranspose         | -                                  | -                                                | yes          |
| DepthToSpace          | attributes(blocksize)              | attributes(blocksize,mode)                       | yes(support) |
| Div                   | -                                  | -                                                | yes          |
| Exp                   | -                                  | -                                                | yes          |
| Expand                | not support                        | support                                          | yes          |
| Gemm                  | -                                  | -                                                | yes          |
| GlobalAveragePool     | -                                  | -                                                | yes          |
| GlobalMaxPool         | -                                  | -                                                | yes          |
| InstanceNormalization | -                                  | -                                                | yes          |
| LeakyRelu             | -                                  | -                                                | yes          |
| MaxPool               | -                                  | attributes(ceil\_mode, dilations,storage\_order) | ?            |
| Mul                   | -                                  | -                                                | yes          |
| PRelu                 | -                                  | -                                                | yes          |
| Pad                   | attributes(pads, value)            | inputs(pads, constant_value)                     | yes(support) |
| ReduceL2              | -                                  | -                                                | yes          |
| ReduceMean            | -                                  | -                                                | yes          |
| ReduceSum             | -                                  | -                                                | yes          |
| Relu                  | -                                  | -                                                | yes          |
| Reshape               | -                                  | -                                                | yes          |
| Slice                 | attributes(starts,ends,axes,steps) | inputs(starts,ends,axes,steps)                   | yes(support) |
| Softmax               | -                                  | -                                                | yes          |
| Softplus              | -                                  | -                                                | yes          |
| Split                 | -                                  | -                                                | yes          |
| Sub                   | -                                  | -                                                | yes          |
| Tanh                  | -                                  | -                                                | yes          |
| Tile                  | -                                  | -                                                | yes          |
| Transpose             | -                                  | -                                                | yes          |
| Upsample              | Upsample                           | deprecated(弃用了), 使用 Resize 替代             | yes          |

(ps: "-" 代表 onnx 1.2.2 和 onnx 1.6.0 版本的 op 是相同的, 没有修改)
