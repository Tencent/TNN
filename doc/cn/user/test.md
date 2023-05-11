# 测试方法

[English Version](../../en/user/test_en.md)

## 一、编译
参考[安装编译](./compile.md)。
打开测试模型开关：  
* `TNN_TEST_ENABLE:BOOL=ON`  
* 对应device的宏，如`TNN_OPENCL_ENABLE`, `TNN_ARM_ENABLE`,`TNN_HUAWEI_NPU_ENABLE`
* 编译完成后，build目录下会生成测试可执行文件test/TNNTest，可在Linux, 安卓ADB等环境下直接运行

## 二、测试方法使用
### 1. 命令
```
TNNTest
必选参数：
  -mp 模型或模型拓扑文件位置
    - 模型为 TNN Proto + Model 格式时，填写 TNN Proto 文件的绝对或相对路径，Proto 与 Model 需同名且在统一路径下，如：-mp ./model/SqueezeNet/squeezenet_v1.1.tnnproto
    - 模型为 TorchScript 格式时，填写 TorchScript 模型文件的绝对或相对路径
    - 模型为 华为 ATLAS OM 格式时，填写 OM 格式模型文件的绝对或相对路径
  -dt 目标设备类型
    - NAIVE - 在X86或ARM CPU端运行，NAIVE模式下的算子使用c++写成，不包含优化功能，可用于正确性校对
    - X86 - 在X86_64 CPU上运行
    - ARM - 在ARM V7/V8 CPU上运行
    - CUDA - 在英伟达 GPU 上运行
    - OPENCL - 使用 OPENCL 在 GPU 上运行
    - METAL - 使用 METAL 在苹果 GPU 上运行
    - HUAWEI_NPU - 在华为海思 NPU 上运行
    - RKNPU - 在瑞芯微 RK NPU 上运行
    - APPLE_NPU - 在苹果 NPU 上运行
    - ZIXIAO - 在紫霄 NPU 上运行
    - ATLAS - 在华为 ATLAS NPU 上运行
  -nt 网络类型
    - 默认类型 - 当目标设备类型为 X86, ARM, CUDA 等时，保持网络类型为默认即可，后台会自动选择默认的网络类型并运行
    - COREML - 运行苹果 CoreML
    - HUAWEI_NPU - 运行华为海思 NPU
    - OPENVINO - 运行 OpenVINO
    - RKNPU - 运行瑞芯微 RK NPU
    - SNPE - 运行高通 SNPE
    - TRT - 运行英伟达 TensorRT
    - TORCH - 运行 Pytorch TorchScript 网络
    - ZIXIAO - 运行紫霄网络
  -op 输出文件位置
    - 空(默认) - 不输出网络运行结果，一般不推荐
    - ${YOUR_OUTPUT_NAME}.txt - 输出文件的绝对或相对路径，指定输出文件后，TNNTest 会将网络所有输出写到该文件中

```
TNNTest
常用可选参数：
  -pr 网络运行精度
    - AUTO(默认) - 根据目标设备与平台，自动选择适合的默认精度，默认精度可能为 float32 或 float16
    - HIGH - 指定精度为 float32
    - LOW - 指定精度为 float16
  -wc 预热运行次数(不计时)
    - 0(默认值) - 默认从第一次循环即开始计时，一些设备中首次或前几次运行推理速度较慢，跳过前几次推理可以使测速结果更加准确
    - N - 指定后，在计时前额外运行 N 次不计时的前向推理
  -ic 循环次数(计时)
    - 1(默认值) - 默认运行前向一次，推荐验证模型运行结果时使用
    - N - 指定后，运行 N 次计时的前向推理
  -dl 设备号码
    - 0(默认值) - 默认在 0 号设备上运行
    - 手动指定 - 可以指定单个或者多个设备，用逗号分割，如 -dl "0,1,2,3,4,5"
  -is 模型输入形状
    - 空(默认值) - 模型为 TNN Proto 格式时从 TNN Proto 文件中读取默认输入形状，其他部分模型格式此项为必选，需手动指定
    - 手动指定 - 手动指定单个或多个输入的形状，用分号分割，如："in_0:1,3,512,512;in_2:1,3,256,256;"
  -it 模型输入数据类型
    - 空(默认值) - 模型为 TNN Proto 格式时从 TNN Proto 文件中读取默认输入形状，其他部分模型格式此项为必选，需手动指定，一些模型格式下默认为 float32
    - 手动指定 - 手动指定单个或多个输入的数据格式，用分号分割，如："in_0:0;;in_2:3;"
    - 数字与数据类型对应关系 - 0->float32; 1->float16; 2->int8; 3->int32; 4->bfp16; 5->int64; 6->uint32; 8->uint8

测试会输出模型耗时：time cost: min = xx   ms  |  max = xx   ms  |  avg = xx   ms

也可作为benchmark工具使用，使用时需要制定wc >= 1，因为第一次运行会准备内存、上下文等增加时间消耗

```
P.S. 华为NPU
NPU需要把HiAI so动态库push到手机上，并将他们添加到LD_LIBRARY_PATH环境变量中.
可以参考 TNN/platform/android/test_android.sh 运行TNNTest

## 三、测试示例
### 1. CUDA 端 TNNTest 示例
```
./scripts/build_cuda_linux/test/TNNTest \
    -mp ./model/SqueezeNet/squeezenet_v1.1.tnnproto \
    -pr HIGH \
    -is "data:1,3,224,224;" \
    -dt CUDA \
    -wc 0 \
    -ic 1 \
    -op ./squeezenet_result.txt
