# NCNN Interface and Model Usage 

[中文版本](../../cn/user/ncnn.md)

## ncnn model usage documentation

When using the ncnn model, you need to specify the ModelType as MODEL_TYPE_NCNN in NetworkConfig parameters.

Reference code:
    
    ModelConfig model_config;
    model_config.model_type = MODEL_TYPE_NCNN;
    TNN net;
    Status ret = net.Init (model_config);
    auto instance = net.CreateInst (network_config, ret);

The default InputShape needs to be declared in TNN when created, which is explained in the Input layer of ncnn.param. If not specified, the InputShape needs to be specified in the Instance creation code.
Reference code:

    InputShapesMap input_shape;
    input_shape ["input_name"] = {1, 3, 224, 224};
    auto instance = net.CsreateInst (network_config, ret, input_shape);

For other aspects, it is the same as the normal flow, and you can refer to other documents for more details.
ss
[Demo example](./demo_en.md) Set the macro TNN_SDK_USE_NCNN_MODEL in examples/samples/TNNSDKSample.h to 1 to run the ncnn model.


### Currently adapted NCNN Op

The current supported NCNN OPs are as follows. Int8 model adaptation is still in progress.

| Operators                  |    NCNN   |   TNN   |
|----------------------------|-----------|---------|
|MemoryData                  |           |    ❌     |
|AbsVal                    sample  |           |    ✅     |
|ArgMax                      | TODO      |         |
|BatchNorm                   |           |    ✅     |
|Bias                        |           |         |
|BinaryOp                    |           |    ✅     |
|BNLL                        |           |         |
|Cast                        |           |         |
|Clip                        |           |         |
|Concat                      |           |    ✅     |
|Convolution                 |           |    ✅     |
|ConvolutionDepthWise        |           |    ✅     |
|Crop                        |           |    ✅     |
|Deconvolution               |           |    ✅     |
|DeconvolutionDepthWise      |           |    ✅     |
|Dequantize                  |           |         |
|DetectionOutput             |           |         |
|Dropout                     |           |    ✅     |
|Eltwise                     |           |         |
|ELU                         |           |         |
|Embed                       |           |         |
|Exp                         |           |         |
|ExpandDims                  |           |         |
|Flatten                     |           |    ✅     |
|HardSigmoid                 |           |    ✅     |
|HardSwish                   |           |    ✅     |
|InnerProduct                |           |    ✅     |
|Input                       |           |    ✅     |
|InstanceNorm                |           |    TODO     |
|Interp                      |           |    ✅     |
|Log                         |           |         |
|LRN                         |           |    ✅     |
|MVN                         |           |         |
|Noop                        |           |         |
|Normalize                   |           |    TODO    |
|Packing                     |           |         |
|Padding                     |           |    TODO     |
|Permute                     |           |    ✅     |
|Pooling                     |           |    ✅     |
|Power                       |           |    TODO     |
|PReLU                       |           |    TODO     |
|PriorBox                    |           |         |
|Proposal                    |           |         |
|PSROIPooling                |           |         |
|Quantize                    |           |         |
|Reduction                   |           |         |
|ReLU                        |           |    ✅     |
|Reorg                       |           |    TODO     |
|Requantize                  |           |         |
|Reshape                     |           |    ✅     |
|ROIAlign                    |           |         |
|ROIPooling                  |           |         |
|Scale                       |           |         |
|SELU                        |           |         |
|ShuffleChannel              |           |    ✅     |
|Sigmoid                     |           |    ✅     |
|Slice                       |           |    ✅     |
|Softmax                     |           |    ✅     |
|Split                       |           |    ✅     |
|SPP                         | TODO      |         |
|Squeeze                     |           |         |
|TanH                        |           |    ✅     |
|Threshold                   |           |         |
|Tile                        | TODO      |         |
|UnaryOp                     |           |         |
|RNN                         | TODO      |         |
|LSTM                        | TODO      |         |


## ncnn model usage documentation

TODO interface is under development