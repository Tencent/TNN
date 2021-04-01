Todo: 详细描述模型及OP支持情况, 包括不同加速平台的支持情况以及包括OP在不同框架间的对应关系
# 支持网络

[English Version](../../en/user/support_en.md)

目前 TNN 支持常用的 CNN 、LSTM 和 BERT 等网络：
- Classical CNN: Vgg AlexNet GoogleNet(v1,v2,v3)
- Practical CNN: ResNet DenseNet SENet
- Light-weight CNN: SqueezeNet MobileNet(v1,v2,v3) ShuffleNet(v1,v2) MNasNet
- Detection: Mtcnn-v2
- Detection: Vgg-ssd SqueezeNet-ssd MobileNetv2-SSDLite ...
- Detection: Yolo-v2 MobileNet-YOLOV3 ...
- Segmentation: FCN PSPNet
- 3D CNN: C3D T3D
- BERT: BERT-Base BERT-Squad MobileBERT DistilBERT
- LSTM: Crnn-LSTM

| model name                | onnx2tnn | Naive | armv7 | armv8 | opencl | metal | Huawei_Npu |
|---------------------------|----------|-----|-------|-------|--------|-------|------|
| AlexNet                   | yes      | yes |       |       |        |       |      |
| DenseNet(121)             | yes      | yes |       |       |        |       |      |
| FCN                       | Yes      | yes |       |       |        |       |      |
| GoogleNet-v1              | yes      | yes |       |       |        |       |      |
| GoogleNet-v2              | yes      | yes |       |       |        |       |      |
| GoogleNet-v3(inception)   | yes      | yes |       |       |        |       |      |
| MnasNet                   | yes      | yes |       |       |        |       |      |
| MobileNet-v1-ssd(caffe)   | yes      | yes | -     | -     | -      | -     |      |
| MobileNet-v1-ssd(pytorch) | yes      | yes |       |       |        |       |      |
| MobileNet-v2-SSDLite      | yes      | yes |       |       |        |       |      |
| MobileNet-yolov3          | ?        | ?   |       |       |        |       |      |
| MobileNet-v1              | yes      | yes |       |       |        |       |      |
| MobileNet-v2              | yes      | yes |       |       |        |       |      |
| MobileNet-v3(small,large) | yes      | yes |       |       |        |       |      |
| Mtcnn-v2                  | yes      | yes |       |       |        |       |      |
| PSPNet                    | yes      | yes |       |       |        |       |      |
| ResNet50                  | yes      | yes |       |       |        |       |      |
| SENet(154)                | yes      | yes |       |       |        |       |      |
| ShuffleNet-v1             | yes      | yes |       |       |        |       |      |
| ShuffleNet-v2             | yes      | yes |       |       |        |       |      |
| SqueezeNet-ssd            | No       | -   | -     | -     | -      | -     |      |
| SqueezeNet-v1             | yes      | yes |       |       |        |       |      |
| UNet                      | yes      | yes |       |       |        |       |      |
| Vgg-ssd                   | yes      | yes |       |       |        |       |      |
| Vgg16                     | yes      | yes |       |       |        |       |      |
| Yolo-v3-tiny              | yes      | yes |       |       |        |       |      |
| Yolo-v2                   | ?        | ?   |       |       |        |       |      |
| Yolo-v2-tiny              | yes      | yes |       |       |        |       |      |
| Yolo-v3                   | yes      | yes |       |       |        |       |      |
| Yolo-v5s                  | yes      | yes |       |       |        |       |      |
| C3D                       | yes      | yes | -     | -     | -      | -     | -    |
| T3D                       | yes      | yes | -     | -     | -      | -     | -    |
| BERT-Base                 | yes      | yes | -     | -     | -      | -     | -    |
| BERT-Squad                | yes      | yes | -     | -     | -      | -     | -    |
| MobileBERT                | yes      | yes | -     | -     | -      | -     | -    |
| DistilBERT                | yes      | yes | -     | -     | -      | -     | -    |
| Crnn-LSTM                 | yes      | yes |       |       |        |       | -    |


1. 关于 upsample 的计算,当参数mode == "bilinear" 或者 mode == "linear", pytorch 转化出的 onnx 模型是有问题的，pytorch 和 onnx 的计算结果是不对齐的。这是 onnx 本身的 bug，这一点尤其需要注意。但是遇到这种情况请不要担心，将转换后的 ONNX 模型转换为 TNN 后，我们保证了 TNN 和 Pytorch 的计算结果是对齐的。经过测试发现会出现上述问题的网络模型有 FCN 以及 PSPNet。
2. 上述表格中的 "?" 表示未知，由于暂时找不到相应类型的模型，所以暂时无法就该模型的兼容性进行测试。

# 支持OP 

| TNN Operators            | Original Operators                             | Naive | armv7 | armv8 | opencl | metal | Huawei_Npu |
|--------------------------|------------------------------------------------|-----|-------|-------|--------|-------|----- |
| Abs                      | Abs                                            | yes | yes   | yes   | yes    | yes   | yes  |
| Acos                     | Acos                                           | yes |       |       | yes    | yes   | yes  |
| Add                      | Add                                            | yes | yes   | yes   | yes    | yes   | yes  |
| ArgMaxOrMin(ArgMax)      | ArgMax                                         | yes |       |       |        |       |      |
| ArgMaxOrMin(ArgMin)      | ArgMin                                         | yes |       |       |        |       |      |
| Asin                     | Asin                                           | yes |       |       | yes    | yes   | yes  |
| Atan                     | Atan                                           | yes |       |       | yes    | yes   |      |
| BatchNormCxx             | BatchNormalization                             | yes | yes   | yes   | yes    | yes   | yes  |
| BitShift                 | BitShift                                       | yes |       |       |        |       |      |
| Cast                     | Cast                                           |     |       |       |        |       |      |
| Ceil                     | Ceil                                           | yes |       |       |        | yes   | yes  |
| Clip                     | Clip                                           | yes | yes   | yes   | yes    | yes   |      |
| Concat                   | Concat                                         | yes | yes   | yes   | yes    | yes   | yes  |
| Const                    | Constant                                       |     |       |       |        |       |      |
| ConstantOfShape          | ConstantOfShape                                | yes |       |       |        |       |      |
| Convolution              | Conv                                           | yes | yes   | yes   | yes    | yes   | yes  |
| Convolution(depthwise)   | Conv                                           | yes | yes   | yes   | yes    | yes   | yes  |
| Convolution(group)       | Conv                                           | yes | yes   | yes   | yes    | yes   | yes  |
| Convolution1D            | Conv                                           | yes |       |       |        |       |      |
| Convolution1D(depthwise) | Conv                                           | yes |       |       |        |       |      |
| Convolution1D(group)     | Conv                                           | yes |       |       |        |       |      |
| Convolution3D            | Conv                                           | yes |       |       |        |       |      |
| Convolution3D(depthwise) | Conv                                           | yes |       |       |        |       |      |
| Convolution3D(group)     | Conv                                           | yes |       |       |        |       |      |
| Cos                      | Cos                                            | yes |       |       | yes    | yes   | yes  |
| Deconvolution            | ConvTranspose                                  | yes | yes   | yes   | yes    | yes   |      |
| Deconvolution(depthwise) | ConvTranspose                                  | yes | yes   | yes   | yes    | yes   |      |
| Deconvolution(group)     | ConvTranspose                                  | yes | yes   | yes   | yes    | yes   |      |
| DetectionOutput          | DectectionOutput(custom operator)              | yes |       |       |        |       |      |
| Div                      | Div                                            | yes | yes   | yes   | yes    | yes   | yes  |
| Dropout                  | Dropout                                        |     |       |       |        |       |      |
| Einsum                   | Einsum                                         | yes |       |       |        |       |      |
| Elu                      | Elu                                            | yes | yes   | yes   | yes    | yes   | yes  |
| Equal                    | Equal                                          | yes |       |       |        |       |      |
| Erf                      | Erf                                            | yes |       |       |        |       |      |
| Exp                      | Exp                                            | yes | yes   | yes   | yes    | yes   | yes  |
| Expand                   | Expand                                         | yes |       |       |        |       |      |
| Flatten                  | Flatten                                        |     |       |       |        |       |      |
| Flatten                  | Shape+Gather+Constant+Unsqueeze+Concat+Reshape |     |       |       |        |       |      |
| Floor                    | Floor                                          | yes |       |       | yes    | yes   | yes  |
| Gather                   | Gather                                         |     |       |       |        |       |      |
| GatherND                 | GatherND                                       | yes |       |       |        |       |      |
| GridSample               | GridSample(PyTorch)                            | yes |       |       |        |       |      |
| GroupNorm                | GroupNorm(PyTorch)                             | yes |       |       |        |       |      |
| HardSigmoid              | HardSigmoid                                    | yes | yes   | yes   | yes    | yes   | yes  |
| HardSwish                | Add + Clip + Div + Mul                         | yes | yes   | yes   | yes    | yes   |      |
| HardSwish                | Add + Clip + Mul + Div                         | yes | yes   | yes   | yes    | yes   |      |
| HardSwish                | HardSigmoid + Mul                              | yes | yes   | yes   | yes    | yes   |      |
| InnerProduct             | Gemm                                           | yes | yes   | yes   | yes    | yes   | yes  |
| InstBatchNormCxx         | InstanceNormalization                          | yes | yes   | yes   | yes    | yes   | yes  |
| Inverse                  | Inverse(PyTorch)                               | yes | yes   | yes   | yes    | yes   | yes  |
| LSTMONNX                 | LSTM                                           | yes |       |       |        |       |      |
| LRN                      | LRN                                            | yes |       |       |        |       | yes  |
| Log                      | Log                                            | yes |       |       | yes    | yes   | yes  |
| LogSigmoid               | Sigmoid + Log                                  | yes | yes   | yes   | yes    | yes   |      |
| MatMul                   | Matmul                                         |     |       |       |        |       |      |
| Max                      | Max                                            | yes | yes   | yes   | yes    | yes   | yes  |
| Min                      | Min                                            | yes | yes   | yes   | yes    | yes   | yes  |
| Mul                      | Mul                                            | yes | yes   | yes   | yes    | yes   | yes  |
| Neg                      | Neg                                            | yes | yes   | yes   | yes    |       | yes  |
| NonZero                  | NonZero                                        | yes |       |       |        |       |      |
| Normalize                | ReduceL2+Clip+Shape+Expand+Div                 | yes | yes   | yes   | yes    | yes   |      |
| Normalize                | Reduce + Clip + Expand + Div                   | yes | yes   | yes   | yes    | yes   |      |
| Normalize                | Mul(square)+Reduce+Max+Sqrt+Mul                | yes | yes   | yes   | yes    | yes   |      |
| OneHot                   | OneHot                                         | yes |       |       |        |       |      |
| PRelu                    | LeakyRelu / PRelu                              | yes | yes   | yes   | yes    | yes   | yes  |
| Pad                      | Pad                                            | yes | yes   | yes   | yes    | yes   | yes  |
| Permute                  | Transpose                                      | yes | yes   | yes   | yes    |       |      |
| PixelShuffle             | PixelShuffle(PyTorch), Depth2Space(ONNX)       | yes |       |       |        |       |      |
| Pooling (Avg)            | AveragePool                                    | yes | yes   | yes   | yes    | yes   | yes  |
| Pooling (GlobalAverage)  | GlobalAveragePool                              | yes | yes   | yes   | yes    | yes   | yes  |
| Pooling (GlobalMax)      | GlobalMaxPool                                  | yes | yes   | yes   | yes    | yes   | yes  |
| Pooling (Max)            | MaxPool                                        | yes | yes   | yes   | yes    | yes   | yes  |
| Pooling3D (Avg)          | AveragePool                                    | yes |       |       |        |       |      |
| Pooling3D (GlobalAverage)| GlobalAveragePool                              | yes |       |       |        |       |      |
| Pooling3D (GlobalMax)    | GlobalMaxPool                                  | yes |       |       |        |       |      |
| Pooling3D (Max)          | MaxPool                                        | yes |       |       |        |       |      |
| Power                    | Pow                                            | yes | yes   | yes   | yes    | yes   |      |
| PriorBox                 | PriorBox(custom operator)                      | yes |       |       | yes    |       | yes  |
| Range                    | Range                                          | yes |       |       |        |       |      |
| Reciprocal               | Reciprocal                                     | yes |       |       | yes    | yes   | yes  |
| ReduceL1                 | ReduceL1                                       | yes |       |       |        |       |      |
| ReduceL2                 | ReduceL2                                       | yes |       |       |        | yes   |      |
| ReduceLogSum             | ReduceLogSum                                   | yes |       |       |        | yes   |      |
| ReduceLogSumExp          | ReduceLogSumExp                                | yes |       |       |        | yes   |      |
| ReduceMax                | ReduceMax                                      | yes |       |       | yes    | yes   |      |
| ReduceMean               | ReduceMean                                     | yes | yes   | yes   | yes    | yes   |      |
| ReduceMin                | ReduceMin                                      | yes |       |       |        | yes   |      |
| ReduceProd               | ReduceProd                                     | yes |       |       |        | yes   | yes  |
| ReduceSum                | ReduceSum                                      | yes | yes   | yes   | yes    | yes   | yes  |
| ReduceSumSquare          | ReduceSumSquare                                | yes |       |       |        | yes   |      |
| Relu                     | Relu                                           | yes | yes   | yes   | yes    | yes   | yes  |
| Relu6                    | Clip                                           | yes | yes   | yes   | yes    | yes   | yes  |
| Reorg                    | DepthToSpace                                   | yes |       |       | yes    |       |      |
| Reorg                    | SpaceToDepth                                   | yes |       |       | yes    |       |      |
| Repeat                   | Tile                                           |     |       |       |        |       |      |
| Reshape                  | Reshape                                        | yes | yes   | yes   | yes    | yes   | yes  |
| RoiAlign                 | RoiAlign                                       | yes |       |       |        |       |      |
| Rsqrt                    | Rsqrt(TFLite)                                  | yes |       |       |        |       |      |
| ScatterND                | ScatterND                                      | yes |       |       |        |       |      |
| Selu                     | Selu                                           | yes |       |       | yes    | yes   | yes  |
| Shape                    | Shape                                          | yes |       |       |        |       |      |
| ShuffleChannel           | Reshape + Transpose + Reshape                  | yes | yes   | yes   | yes    | yes   | yes  |
| Sigmoid                  | Sigmoid                                        | yes | yes   | yes   | yes    | yes   | yes  |
| Sign                     | Sign                                           | yes | yes   | yes   | yes    | yes   | yes  |
| SignedMul                | Sub+Sign+Add+Div+Gather+Slice+Mul              |     |       |       |        |       |      |
| Sin                      | Sin                                            | yes |       |       | yes    | yes   |  yes |
| Size                     | Size                                           | yes |       |       |        |       |      |
| Slice(StrideSlice)       | Slice                                          | yes | yes   | yes   | yes    | yes   |  yes |
| Softmax                  | Softmax                                        | yes | yes   | yes   | yes    | yes   |  yes |
| Softplus                 | Softplus                                       | yes | yes   | yes   | yes    | yes   |  yes |
| Softsign                 | Softsign                                       | yes |       |       |        |       |      |
| Split                    | Split                                          |     |       |       | yes    |       |      |
| Sqrt                     | Sqrt                                           | yes | yes   | yes   | yes    | yes   | yes  |
| SquaredDifference        | SquaredDifference(TFLite)                      | yes |       |       |        |       |      |
| Squeeze                  | Squeeze                                        |     |       |       |        |       |      |
| Sub                      | Sub                                            | yes | yes   | yes   | yes    | yes   | yes  |
| Sum                      |                                                |     |       |       |        |       |      |
| Tan                      | Tan                                            | yes |       |       | yes    | yes   | yes  |
| Tanh                     | Tanh                                           | yes | yes   | yes   | yes    | yes   | yes  |
| Tile                     | Tile                                           | yes |       |       |        |       |      |
| Unsqueeze                | Unsqueeze                                      | yes |       |       |        |       |      |
| Upsample                 | Upsample / Resize                              | yes | yes   | yes   | yes    | yes   | yes  |
| Where                    | Where                                          | yes |       |       |        |       |      |



1. 在上面的表格中 TNN 的HardSwish 算子对应 ONNX 算子是"Add + Clip + Div + Mul"，这代表 ONNX 的四个算子共同组合成了 TNN 中的 HardSwsh 算子。表格中的 "+" 符号代表算子的组合，其他类似的算子同理。
2. 在上面的表格中 TNN 的 PRelu 算子对应着 ONNX 的"LeakyRelu / PRelu"，这代表 TNN 中的 RRelu 算子同时支持 ONNX 中的 PRelu 和 LeakyRelu 算子。


# 支持硬件

| device  | support |
|-------- |---------|
| ARMv7   |  Yes    |
| ARMv8   |  Yes    |
| OpenCL  |  Yes    |
| Metal   |  Yes    |
| 华为Npu |  Yes    |
| RK NPU  |  Yes    |
| X86     |  Yes    |
| CUDA    |  Yes    |


1. 华为NPU仅支持达芬奇架构NPU，目前有：麒麟810，麒麟820，麒麟985，麒麟990，麒麟990 5G，麒麟990E，麒麟9000，麒麟9000E等。
