# onnx version

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
    
