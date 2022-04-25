<div align=left ><img src="https://github.com/darrenyao87/tnn-models/raw/master/TNN.png"/>

# Run a demo from scratch 

[中文版本](../cn/get_started.md)

It is very simple to use TNN. If you have a trained model, the model can be deployed on the target platform through three steps.
1. Convert the trained model into a TNN model. We provide a wealth of tools to help you complete this step, whether you are using Tensorflow, Pytorch, or Caffe, you can easily complete the conversion.
Detailed hands-on tutorials can be found here [How to Create TNN Models](./user/convert_en.md).

2. When you have finished converting the model, the second step is to compile the TNN engine of the target platform. You can choose among different acceleration solutions such as ARM/OpenCL/Metal/Npu according to the hardware support.
   For these platforms, TNN provides convenient one-click scripts to compile. For detailed steps, please refer to [How to Compile TNN](./user/compile_en.md).

3. The final step is to use the compiled TNN engine for inference. You can make program calls to TNN inside your application. We provide a rich and detailed demo as a reference to help you complete.
    * [Run an iOS Demo](./user/demo_en.md)
    * [Run an Android Demo](./user/demo_en.md)
