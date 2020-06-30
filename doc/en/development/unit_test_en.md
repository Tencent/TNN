# Unit Test

[中文版本](../../cn/development/unit_test.md)

This document mainly introduces the purpose, usage, and note of the unit test.

## Why Unit Test

The current unit test has two purposes:

1. Verify the correctness of each OPs' results on different platforms.
2. Act as an OP performance testing tool to test OP performance without the need for constructing a whole new model.

## How Unit Test Works

The OP in the TNN code is implemented by the Layer class, but the Layer class only provides platform-independent logic such as Blob Shape inference. The actual operator implementations for different platforms are implemented in layer_acc.
Therefore, the Layer unit test is calculated by comparing the results from two platforms, and then the correctness is checked.

## Instructions

### How to compile

Turn on the TNN_UNIT_TEST_ENABLE switch when compiling:

    cmake -DTNN_UNIT_TEST_ENABLE = ON ../
    
If it is used for OP performance test, you need to turn on the TNN_BENCHMARK_ENABLE switch at the same time:

    cmake -DTNN_UNIT_TEST_ENABLE = ON -DTNN_BENCHMARK_ENABLE = ON ../


* Turn on the following options to compile TNN (For the compilation method. Please refer to [Compile TNN](../user/compile_en.md))
* TNN_UNIT_TEST_ENABLE = ON
* You need to enable TNN_BENCHMARK_ENABLE at the same time for OP performance test:
* TNN_BENCHMARK_ENABLE = ON


### How to run

After successful compilation, execute the following command:

    ./test/unit_test/unit_test -ic 1
    
The ic parameter is used to control the number of times each unit test is repeated, usually 1 is sufficient. Other optional parameters are as follows:

    -dt {ARM | OPENCL | METAL} // Type of computing device tested
    -lp $ {load_library_path} // Library path to be loaded by OPENCL and METAL
    -th $ {num_threads} // number of threads, default is 1
    -ub {0 | 1} // Whether to print the calculation performance data (GFLOPS) for performance testing
    
An actual test example shows below:
    
    ./test/unit_test/unit_test -ic 1 -dt ARM -th 4 -ub 0
    

## Note 

In the unit test, many parameter combinations are generated through the GTEST WithParamInterface interface. If you need to change or customize the parameters, you can take a look at the INSTANTIATE_TEST_SUITE_P macro.