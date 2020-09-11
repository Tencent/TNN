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
    -mp 模型proto位置(模型model需要在同文件夹下同前缀名)
    -dt DEVICE类型（ARM, OPENCL, HUAWEI_NPU）
常用可选参数：
    -nt network类型（默认naive， 华为Npu需要特殊指定 -nt HUAWEI_NPU）
    -op 输出文件位置   
    -ic 循环次数  
    -wc warmup运行次数
    -dl 设备list
    -ip 输入文件
    -it（输入类型，默认为NCHW float）
    -th (CPU线程数)  

测试会输出模型耗时：time cost: min = xx   ms  |  max = xx   ms  |  avg = xx   ms

也可作为benchmark工具使用，使用时需要制定wc >= 1，因为第一次运行会准备内存、上下文等增加时间消耗

```
P.S. 华为NPU
NPU需要把HiAI so动态库push到手机上，并将他们添加到LD_LIBRARY_PATH环境变量中.
可以参考 TNN/platform/android/test_android.sh 运行TNNTest
 