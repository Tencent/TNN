# 单元测试 

[English Version](../../en/development/unit_test_en.md)

本文档主要介绍当前单元测试的目的、用法及注意事项。

##  单元测试用途

当前单元测试有两个用途：

1. 验证各个OP在不同平台上的结果正确性。
2. 充当OP性能测试工具，在不需要模型的情况下测试OP性能。

## 需了解的代码的信息

TNN代码中OP通过Layer这个类型来实现，但Layer类型仅仅实现了Blob Shape推理等计算无关的逻辑。不同平台的计算由layer_acc实现。
因此，Layer单元测试中通过两个层计算，然后对比结果，以此对比结果正确性。

## 使用方法

### 编译方法

* 打开以下选项编译TNN（编译方法参照[TNN编译文档](../user/compile.md))
* TNN_UNIT_TEST_ENABLE=ON 
* 如果用于OP性能测试，需同时打开 TNN_BENCHMARK_ENABLE 开关:
* TNN_BENCHMARK_ENABLE=ON 
    
### 运行方法

编译成功后执行以下命令：

    ./test/unit_test/unit_test -ic 1
    
ic 参数用于控制每个单元测试重复进行的次数，通常用1即可，其他可选参数如下:

    -dt {ARM|OPENCL|METAL} // 测试的计算设备类型
    -lp ${load_library_path} // OPENCL 及 METAL 需要加载的库路径
    -th ${num_threads} // 线程数，默认为1
    -ub {0|1} // 是否打印计算性能数据(GFLOPS)，用于性能测试
    
一个实际的测试例子如下:
    
    ./test/unit_test/unit_test -ic 1 -dt ARM -th 4 -ub 0
    

## 注意事项 

单元测试中通过GTEST WithParamInterface 接口生成了很多参数组合。若需更改或自定义参数，可查看 INSTANTIATE_TEST_SUITE_P 宏相关代码。

