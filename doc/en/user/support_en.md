# TNN Support

[中文版本](../../cn/user/support.md)

## TNN supported models

TNN currently support main-stream CNN, LSTM and BERT networks：
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

| model name                | onnx2tnn | Naive | armv7 | armv8 | opencl | metal | Huawei_Npu | CUDA | x86 | OpenVINO | RKNPU | Apple NPU |
|---------------------------|----------|-----|-------|-------|--------|-------|-----|------|------|------|------|------|
| AlexNet                   | yes      | yes | -     | -     |        | yes   | yes | yes  | yes  | yes  |      |
| DenseNet(121)             | yes      | yes | yes   | yes   | yes    | yes   | yes | yes  | yes  | yes  | yes  | yes  |
| FCN                       | Yes      | yes | yes   | yes   | yes    | yes   |  -  | yes  | yes  | yes  |      |
| GoogleNet-v1              | yes      | yes | yes   | yes   |        |       |     |      | yes  | yes  |      |
| GoogleNet-v2              | yes      | yes | yes   | yes   |        |       |     |      | yes  | yes  |      | yes  |
| GoogleNet-v3(inception)   | yes      | yes | yes   | yes   |        |       |     |      | yes  | yes  |      | yes  |
| MnasNet                   | yes      | yes |       |       |        |       |     |      | yes  | yes  |      |
| MobileNet-v1-ssd(caffe)   | yes      | yes | -     | -     | -      | -     |  -  |  -   | yes  | yes  |      |
| MobileNet-v1-ssd(pytorch) | yes      | yes | yes   | yes   | yes    | yes   | yes | yes  | yes  | yes  |      |
| MobileNet-v2-SSDLite      | yes      | yes | yes   | yes   | yes    | yes   | yes | yes  | yes  | yes  |      |
| MobileNet-yolov3          | ?        | ?   |       |       |        |       |     |      |      |      |      |
| MobileNet-v1              | yes      | yes | yes   | yes   | yes    | yes   | yes | yes  | yes  | yes  | yes  | yes  |
| MobileNet-v2              | yes      | yes | yes   | yes   | yes    | yes   | yes | yes  | yes  | yes  | yes  | yes  |
| MobileNet-v3(small,large) | yes      | yes | yes   | yes   | yes    | yes   | No  | yes  | yes  | yes  |      |
| Mtcnn-v2                  | yes      | yes | yes   | yes   | yes    | yes   | yes | yes  | yes  | yes  |      |
| PSPNet                    | yes      | yes | yes   | yes   | yes    | yes   | No  | yes  | yes  | yes  |      |
| ResNet50                  | yes      | yes | yes   | yes   | yes    | yes   | yes | yes  | yes  | yes  |      | yes  |
| SENet(154)                | yes      | yes | yes   | yes   | yes    | yes   |  -  | yes  | yes  | yes  |      |
| ShuffleNet-v1             | yes      | yes | yes   | yes   | yes    | yes   | yes | yes  | yes  | yes  |      | yes  |
| ShuffleNet-v2             | yes      | yes | yes   | yes   | yes    | yes   | yes | yes  | yes  | yes  |      | yes  |
| SqueezeNet-ssd            | No       | -   | -     | -     | -      | -     |  -  |  -   | -    | -    |      |
| SqueezeNet-v1             | yes      | yes | yes   | yes   | yes    | yes   | yes | yes  | yes  | yes  | yes  | yes  |
| UNet                      | yes      | yes | yes   | yes   | yes    | yes   | yes | yes  | yes  | yes  |      | yes  |
| Vgg-ssd                   | yes      | yes | yes   | yes   | yes    | yes   | yes | yes  | yes  | yes  |      |
| Vgg16                     | yes      | yes | yes   | yes   |        | yes   | yes |      |      |      |      |
| Yolo-v3-tiny              | yes      | yes | yes   | yes   | yes    | yes   | yes | yes  | yes  | yes  |      |
| Yolo-v2                   | ?        | ?   | yes   | yes   | yes    | yes   | yes |      |      |      |      |
| Yolo-v2-tiny              | yes      | yes | yes   | yes   | yes    | yes   | yes | yes  | yes  | yes  |      | yes  |
| Yolo-v3                   | yes      | yes | yes   | yes   | yes    | yes   | -   | yes  | yes  | yes  |      |
| Yolo-v5s                  | yes      | yes | yes   | yes   | yes    | yes   | yes |      | yes  | yes  |      | yes  |
| C3D                       | yes      | yes | -     | -     | -      | -     | -   |      | -    | -    |      |
| T3D                       | yes      | yes | -     | -     | -      | -     | -   |      | -    | -    |      |
| BERT-Base                 | yes      | yes | -     | -     | -      | -     | -   |      |      |      |      |
| BERT-Squad                | yes      | yes | -     | -     | -      | -     | -   |      |      |      |      |
| MobileBERT                | yes      | yes | -     | -     | -      | -     | -   | yes  |      |      |      |
| DistilBERT                | yes      | yes | -     | -     | -      | -     | -   |      |      |      |      |
| Crnn-LSTM                 | yes      | yes | yes   | yes   | yes    | yes   | -   | yes  | yes  | yes  |      |


1. Regarding the upsample calculation of upsample, when the parameter mode == "bilinear" or mode == "linear", the onnx model exported by pytorch has some issues, and the calculation results of pytorch and onnx are not aligned. This is a bug of onnx itself, which deserves special attention. But don't worry about this problem. After converting the converted ONNX model to TNN, we ensure that the calculation results of TNN and Pytorch are aligned. In our testing, FCN and PSPNet have such problems.
2. The "?" In the above table means unknown. Since the corresponding type of the model cannot be found, the compatibility of the model cannot be tested now.

## TNN supported operators

| TNN Operators            | Original Operators                             | Naive | armv7 | armv8 | opencl | metal | Huawei_Npu | CUDA | x86 | OpenVINO | RKNPU | Apple NPU |
|--------------------------|------------------------------------------------|-----|-------|-------|--------|-------|------|-------|-------|-------|------|------|
| Abs                      | Abs                                            | yes | yes   | yes   | yes    | yes   | yes  | yes   | yes   | yes   | yes  |
| Acos                     | Acos                                           | yes | yes   | yes   | yes    | yes   | yes  | yes   | yes   | yes   |      |
| Add                      | Add                                            | yes | yes   | yes   | yes    | yes   | yes  | yes   | yes   | yes   |      | yes  |
| ArgMaxOrMin(ArgMax)      | ArgMax                                         | yes | yes   | yes   | yes    | yes   | yes  | yes   | yes   | yes   |      |
| ArgMaxOrMin(ArgMin)      | ArgMin                                         | yes | yes   | yes   | yes    | yes   |      | yes   | yes   | yes   |      |
| Asin                     | Asin                                           | yes | yes   | yes   | yes    | yes   | yes  | yes   | yes   | yes   |      |
| Atan                     | Atan                                           | yes | yes   | yes   | yes    | yes   | yes  | yes   | yes   | yes   |      |
| BatchNormCxx             | BatchNormalization                             | yes | yes   | yes   | yes    | yes   | yes  | yes   | yes   | yes   | yes  | yes  |
| BitShift                 | BitShift                                       | yes |       |       |        |       |      | yes   |       |       |      |
| Cast                     | Cast                                           | yes | yes   | yes   | yes    | yes   | yes  | yes   | yes   | yes   |      | yes  |
| Ceil                     | Ceil                                           | yes | yes   | yes   | yes    | yes   | yes  | yes   | yes   | yes   |      |
| Clip                     | Clip                                           | yes | yes   | yes   | yes    | yes   | yes  | yes   | yes   | yes   |      | yes  |
| Concat                   | Concat                                         | yes | yes   | yes   | yes    | yes   | yes  | yes   | yes   | yes   | yes  | yes  |
| Const                    | Constant                                       |     |       |       |        |       |      |       |       |       |      | yes  |
| ConstantOfShape          | ConstantOfShape                                | yes |       |       |        |       |      |       |       |       |      |
| Convolution              | Conv                                           | yes | yes   | yes   | yes    | yes   | yes  | yes   | yes   | yes   | yes  | yes  |
| Convolution(depthwise)   | Conv                                           | yes | yes   | yes   | yes    | yes   | yes  | yes   | yes   | yes   | yes  | yes  |
| Convolution(group)       | Conv                                           | yes | yes   | yes   | yes    | yes   | yes  | yes   | yes   | yes   | yes  | yes  |
| Convolution1D            | Conv                                           | yes |       |       |        |       |      | yes   |       |       |      |
| Convolution1D(depthwise) | Conv                                           | yes |       |       |        |       |      | yes   |       |       |      |
| Convolution1D(group)     | Conv                                           | yes |       |       |        |       |      | yes   |       |       |      |
| Convolution3D            | Conv                                           | yes |       |       |        |       |      | yes   |       |       |      |
| Convolution3D(depthwise) | Conv                                           | yes |       |       |        |       |      | yes   |       |       |      |
| Convolution3D(group)     | Conv                                           | yes |       |       |        |       |      | yes   |       |       |      |
| Cos                      | Cos                                            | yes | yes   | yes   | yes    | yes   | yes  | yes   | yes   | yes   |      |
| Deconvolution            | ConvTranspose                                  | yes | yes   | yes   | yes    | yes   | yes  | yes   | yes   | yes   |      | yes  |
| Deconvolution(depthwise) | ConvTranspose                                  | yes | yes   | yes   | yes    | yes   | yes  | yes   | yes   | yes   |      | yes  |
| Deconvolution(group)     | ConvTranspose                                  | yes | yes   | yes   | yes    | yes   | yes  | yes   | yes   | yes   |      |
| DetectionOutput          | DectectionOutput(custom operator)              | yes | yes   | yes   |        |       |      | yes   | yes   | yes   |      |
| Div                      | Div                                            | yes | yes   | yes   | yes    | yes   | yes  | yes   | yes   | yes   |      | yes  |
| Dropout                  | Dropout                                        |     |       |       |        |       |      |       |       |       |      |
| Einsum                   | Einsum                                         | yes |       |       |        |       |      | yes   |       |       |      |
| Elu                      | Elu                                            | yes | yes   | yes   | yes    | yes   | yes  | yes   | yes   | yes   | yes  |
| Equal                    | Equal                                          | yes |       |       |        |       |      |       |       |       |      |
| Erf                      | Erf                                            | yes |       |       |        |       |      | yes   | yes   | yes   |      |
| Exp                      | Exp                                            | yes | yes   | yes   | yes    | yes   | yes  | yes   | yes   | yes   | yes  | yes  |
| Expand                   | Expand                                         | yes | yes   | yes   |        |       |      | yes   | yes   | yes   |      |
| Flatten                  | Flatten                                        | yes | yes   | yes   | yes    | yes   | yes  | yes   | yes   | yes   |      |
| Flatten                  | Shape+Gather+Constant+Unsqueeze+Concat+Reshape |     |       |       |        |       |      |       |       |       |      |
| Floor                    | Floor                                          | yes | yes   | yes   | yes    | yes   | yes  | yes   | yes   | yes   |      |
| Gather                   | Gather                                         | yes | yes   | yes   | yes    | yes   | yes  | yes   | yes   | yes   | yes  | yes  |
| GatherND                 | GatherND                                       | yes |       |       |        |       |      | yes   |       |       |      |
| GridSample               | GridSample(PyTorch)                            | yes |       |       |        |       |      | yes   |       |       |      |
| GroupNorm                | GroupNorm(PyTorch)                             | yes |       |       |        |       |      | yes   |       |       |      |
| HardSigmoid              | HardSigmoid                                    | yes | yes   | yes   | yes    | yes   | yes  | yes   | yes   | yes   |      | yes  |
| HardSwish                | Add + Clip + Div + Mul                         | yes | yes   | yes   | yes    | yes   | yes  | yes   | yes   | yes   |      |
| HardSwish                | Add + Clip + Mul + Div                         | yes | yes   | yes   | yes    | yes   | yes  | yes   | yes   | yes   |      |
| HardSwish                | HardSigmoid + Mul                              | yes | yes   | yes   | yes    | yes   | yes  | yes   | yes   | yes   |      | yes  |
| InnerProduct             | Gemm                                           | yes | yes   | yes   | yes    | yes   | yes  | yes   | yes   | yes   | yes  | yes  |
| InstBatchNormCxx         | InstanceNormalization                          | yes | yes   | yes   | yes    | yes   | yes  | yes   | yes   | yes   |      |
| Inverse                  | Inverse(PyTorch)                               | yes |       |       |        |       |      | yes   |       |       |      |
| LSTMONNX                 | LSTM                                           | yes | yes   | yes   | yes    | yes   |      | yes   | yes   | yes   |      |
| LRN                      | LRN                                            | yes |       |       |        | yes   | yes  | yes   | yes   | yes   |      |
| Log                      | Log                                            | yes | yes   | yes   | yes    | yes   | yes  | yes   | yes   | yes   |      |
| LogSigmoid               | Sigmoid + Log                                  | yes | yes   | yes   | yes    | yes   | yes  | yes   | yes   | yes   |      |
| MatMul                   | Matmul                                         | yes | yes   | yes   | yes    | yes   | yes  | yes   | yes   | yes   |      | yes  |
| Max                      | Max                                            | yes | yes   | yes   | yes    | yes   | yes  | yes   | yes   | yes   | yes  |
| Min                      | Min                                            | yes | yes   | yes   | yes    | yes   | yes  | yes   | yes   | yes   | yes  |
| Mul                      | Mul                                            | yes | yes   | yes   | yes    | yes   | yes  | yes   | yes   | yes   | yes  | yes  |
| Neg                      | Neg                                            | yes | yes   | yes   | yes    | yes   | yes  | yes   | yes   | yes   | yes  |
| NonZero                  | NonZero                                        | yes |       |       |        |       |      |       |       |       |      |
| Normalize                | ReduceL2+Clip+Shape+Expand+Div                 | yes | yes   | yes   | yes    | yes   | yes  | yes   | yes   | yes   |      |
| Normalize                | Reduce + Clip + Expand + Div                   | yes | yes   | yes   | yes    | yes   |      | yes   | yes   | yes   |      |
| Normalize                | Mul(square)+Reduce+Max+Sqrt+Mul                | yes | yes   | yes   | yes    | yes   |      | yes   | yes   | yes   | yes  |
| OneHot                   | OneHot                                         | yes |       |       |        |       |      | yes   |       |       |      |
| PRelu                    | LeakyRelu / PRelu                              | yes | yes   | yes   | yes    | yes   | yes  | yes   | yes   | yes   | yes  | yes  |
| Pad                      | Pad                                            | yes | yes   | yes   | yes    | yes   | yes  | yes   | yes   | yes   | yes  | yes  |
| Permute                  | Transpose                                      | yes | yes   | yes   | yes    | yes   | yes  | yes   | yes   | yes   |      | yes  |
| PixelShuffle             | PixelShuffle(PyTorch), Depth2Space(ONNX)       | yes | yes   | yes   | yes    | yes   |      | yes   | yes   |       | yes  |
| Pooling (Avg)            | AveragePool                                    | yes | yes   | yes   | yes    | yes   | yes  | yes   | yes   | yes   |      | yes  |
| Pooling (GlobalAverage)  | GlobalAveragePool                              | yes | yes   | yes   | yes    | yes   | yes  | yes   | yes   | yes   | yes  | yes  |
| Pooling (GlobalMax)      | GlobalMaxPool                                  | yes | yes   | yes   | yes    | yes   | yes  | yes   | yes   | yes   | yes  | yes  |
| Pooling (Max)            | MaxPool                                        | yes | yes   | yes   | yes    | yes   | yes  | yes   | yes   | yes   | yes  |
| Pooling3D (Avg)          | AveragePool                                    | yes |       |       |        |       |      | yes   |       |       |      |
| Pooling3D (GlobalAverage)| GlobalAveragePool                              | yes |       |       |        |       |      | yes   |       |       |      |
| Pooling3D (GlobalMax)    | GlobalMaxPool                                  | yes |       |       |        |       |      | yes   |       |       |      |
| Pooling3D (Max)          | MaxPool                                        | yes |       |       |        |       |      | yes   |       |       |      |
| Power                    | Pow                                            | yes | yes   | yes   | yes    | yes   | yes  | yes   | yes   |       |      |
| PriorBox                 | PriorBox(custom operator)                      | yes | yes   | yes   | yes    | yes   | yes  | yes   | yes   | yes   |      |
| Range                    | Range                                          | yes |       |       |        |       |      |       |       |       |      |
| Reciprocal               | Reciprocal                                     | yes | yes   | yes   | yes    | yes   | yes  | yes   | yes   | yes   |      |
| ReduceL1                 | ReduceL1                                       | yes | yes   | yes   | yes    | yes   |      |       | yes   | yes   |      |
| ReduceL2                 | ReduceL2                                       | yes | yes   | yes   | yes    | yes   |      | yes   | yes   | yes   |      |
| ReduceLogSum             | ReduceLogSum                                   | yes | yes   | yes   | yes    | yes   |      |       | yes   | yes   |      |
| ReduceLogSumExp          | ReduceLogSumExp                                | yes | yes   | yes   | yes    | yes   | yes  | yes   | yes   | yes   |      |
| ReduceMax                | ReduceMax                                      | yes | yes   | yes   | yes    | yes   | yes  | yes   | yes   | yes   |      | yes  |
| ReduceMean               | ReduceMean                                     | yes | yes   | yes   | yes    | yes   | yes  | yes   | yes   | yes   | yes  | yes  |
| ReduceMin                | ReduceMin                                      | yes | yes   | yes   | yes    | yes   | yes  | yes   | yes   | yes   |      | yes  |
| ReduceProd               | ReduceProd                                     | yes | yes   | yes   | yes    | yes   | yes  |       | yes   | yes   |      | 
| ReduceSum                | ReduceSum                                      | yes | yes   | yes   | yes    | yes   | yes  | yes   | yes   | yes   |      | yes  |
| ReduceSumSquare          | ReduceSumSquare                                | yes | yes   | yes   | yes    | yes   |      |       | yes   | yes   |      |
| Relu                     | Relu                                           | yes | yes   | yes   | yes    | yes   | yes  | yes   | yes   | yes   | yes  | yes  |
| Relu6                    | Clip                                           | yes | yes   | yes   | yes    | yes   | yes  | yes   | yes   | yes   | yes  | yes  |
| Reorg                    | DepthToSpace                                   | yes | yes   | yes   | yes    | yes   |      |       | yes   | yes   |      |
| Reorg                    | SpaceToDepth                                   | yes | yes   | yes   | yes    | yes   |      |       | yes   | yes   |      |
| Repeat                   | Tile                                           |     |       |       |        |       |      |       |       |       |      |
| Reshape                  | Reshape                                        | yes | yes   | yes   | yes    | yes   | yes  | yes   | yes   | yes   |      | yes  |
| RoiAlign                 | RoiAlign                                       | yes |       |       |        |       |      | yes   |       |       |      |
| Rsqrt                    | Rsqrt(TFLite)                                  | yes | yes   | yes   |        |       |      |       |       |       |      |
| ScatterND                | ScatterND                                      | yes |       |       |        |       |      | yes   | yes   | yes   |      |
| Selu                     | Selu                                           | yes | yes   | yes   | yes    | yes   | yes  |       | yes   | yes   |      |
| Shape                    | Shape                                          | yes |       |       |        |       | yes  | yes   |       |       |      | yes  |
| ShuffleChannel           | Reshape + Transpose + Reshape                  | yes | yes   | yes   | yes    | yes   | yes  |       | yes   | yes   |      | yes  |
| Sigmoid                  | Sigmoid                                        | yes | yes   | yes   | yes    | yes   | yes  | yes   | yes   | yes   | yes  | yes  |
| Sign                     | Sign                                           | yes | yes   | yes   | yes    | yes   | yes  | yes   | yes   | yes   |      |
| SignedMul                | Sub+Sign+Add+Div+Gather+Slice+Mul              | yes | yes   | yes   |        |       |      |       | yes   |       |      |
| Sin                      | Sin                                            | yes | yes   | yes   | yes    | yes   | yes  | yes   | yes   | yes   |      |
| Size                     | Size                                           | yes |       |       |        |       |      |       |       |       |      |
| Slice(StrideSlice)       | Slice                                          | yes | yes   | yes   | yes    | yes   | yes  | yes   | yes   | yes   |      | yes  |
| Softmax                  | Softmax                                        | yes | yes   | yes   | yes    | yes   | yes  | yes   | yes   | yes   | yes  | yes  |
| Softplus                 | Softplus                                       | yes | yes   | yes   | yes    | yes   | yes  | yes   | yes   | yes   |      | yes  |
| Softsign                 | Softsign                                       | yes |       |       |        |       | yes  |       | yes   | yes   |      | yes  |
| Split                    | Split                                          |     | yes   | yes   | yes    | yes   | yes  | yes   | yes   | yes   |      | yes  |
| Sqrt                     | Sqrt                                           | yes | yes   | yes   | yes    | yes   | yes  | yes   | yes   | yes   |      |
| SquaredDifference        | SquaredDifference(TFLite)                      | yes |       |       |        |       |      |       |       |       |      |
| Squeeze                  | Squeeze                                        |     | yes   | yes   | yes    | yes   | yes  | yes   | yes   | yes   |      | yes  |
| Sub                      | Sub                                            | yes | yes   | yes   | yes    | yes   | yes  | yes   | yes   | yes   | yes  | yes  |
| Sum                      |                                                |     |       |       |        |       |      |       |       |       |      |
| Tan                      | Tan                                            | yes | yes   | yes   | yes    | yes   | yes  | yes   | yes   | yes   |      | yes  |
| Tanh                     | Tanh                                           | yes | yes   | yes   | yes    | yes   | yes  | yes   | yes   | yes   |      |
| Tile                     | Tile                                           | yes |       |       |        |       |      | yes   |       |       |      |
| Unsqueeze                | Unsqueeze                                      | yes | yes   | yes   | yes    | yes   | yes  | yes   | yes   | yes   |      | yes  |
| Upsample                 | Upsample / Resize                              | yes | yes   | yes   | yes    | yes   | yes  | yes   | yes   | yes   |      | yes  |
| Where                    | Where                                          | yes |       |       |        |       |      |       |       |       |      |


1. In the above table, the TNN HardSwish operator maps to "Add + Clip + Div + Mul" in ONNX, which means that the four operators of ONNX are combined into the HardSwsh operator in TNN. The "+" symbol in the table represents the combination of operators. This applies to other similar operators.
2. In the table above, the PRelu operator of TNN maps to "LeakyRelu / PRelu" in ONNX, which means that the RRelu operator in TNN supports both PRelu and LeakyRelu operators in ONNX.

## TNN supported devices

| device     | support |
|------------|---------|
| ARMv7      |  Yes    |
| ARMv8      |  Yes    |
| OpenCL     |  Yes    |
| Metal      |  Yes    |
| HuaweiNPU  |  Yes    |
| AppleNPU   |  Yes    |
| RKNPU      |  Yes    |
| X86        |  Yes    |
| CUDA       |  Yes    |

1. HuaweiNPU is DaVinci NPU of Huawei, as follows: Kirin810, Kirin820, Kirin985, Kirin990, Kirin990 5G, Kirin990E, Kirin9000, Kirin9000E etc.
2. RockchipNPU only support fp16 mode of rk1808