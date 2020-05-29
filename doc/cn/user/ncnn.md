
# ncnn 接口及模型使用文档

## ncnn 模型使用文档

使用ncnn 模型时，需要在 TNN 初始化参数 NetworkConfig 中指明 ModelType 为 MODEL_TYPE_NCNN。

具体代码参考：
    
    ModelConfig model_config;
    model_config.model_type = MODEL_TYPE_NCNN;
    TNN net;
    Status ret = net.Init(model_config);
    auto instance = net.CreateInst(network_config, ret);

TNN Instance 在创建时需要声明默认InputShape，通常ncnn.param 的Input 层中会说明。如果其中未指明的话，需要在创建Instance 代码中指明。
具体参考: 

    InputShapesMap input_shape;
    input_shape["input_name"] = {1, 3, 224, 224};
    auto instance = net.CreateInst(network_config, ret, input_shape);

其他方面使用与正常调用流程相同，可具体参考其他文档。

[Demo示例](demo.md)中可将examples/samples/TNNSDKSample.h中的宏TNN_SDK_USE_NCNN_MODEL设置为1来运行ncnn模型。



### 当前适配完成的NCNN Op

目前对NCNN OP 支持情况如下, Int8 模型适配还在进行中。

| Operators                  |    NCNN   |   TNN   |
|----------------------------|-----------|---------|
|MemoryData                  |           |    ❌     |
|AbsVal                      |           |    ✅     |
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


## ncnn 模型使用文档

