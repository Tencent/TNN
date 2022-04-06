<div align=left ><img src="https://github.com/darrenyao87/tnn-models/raw/master/TNN.png"/>

# 从0开始跑通一个Demo

[English Version](../en/get_started_en.md)

使用TNN非常简单，如果你有一个已经训练好的模型, 那么一般而言通过以下三个步骤就能完成模型在目标平台上的部署。
1. 第一步是把训练好的模型转换成TNN的模型，为此我们提供了丰富的工具来帮助你完成这一步，无论你使用的是Tensorflow、Pytorch、或者Caffe，都可以轻松完成转换。
详细的手把手教程可以参见这里[如何转换模型](./user/convert.md)。

2. 当你完成了模型的转换，第二步就是编译目标平台的TNN引擎了，你可以根据自己的目标平台的硬件支持情况，选择CPU/ARM/OpenCL/Metal/NPU等加速方案。
   对于这些平台，TNN都提供了一键编译的脚本，使用非常方便。详细步骤可以参考这里[如何编译TNN](./user/compile.md)。

3. 最后一步就是使用编译好的TNN引擎进行推理，你可以在自己的应用程序中嵌入对TNN的调用，这方面我们提供了丰富而详实的demo来帮助你完成。
    *  [从0开始跑通一个iOS Demo](./user/demo.md)
    *  [从0开始跑通一个Android Demo](./user/demo.md)
