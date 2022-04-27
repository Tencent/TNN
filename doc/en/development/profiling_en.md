# Model Performance Analysis

[中文版本](../../cn/development/profiling.md)

Analyze the running time of a model.

## I. Test the time cost on iOS platform
### Test Steps
1. Add a test model

  Add test models in the `<path_to_tnn>/model` directory, each model has a folder, and the folder contains model files ending with `proto` and `model`. There is already a model `squeezenetv1.1` in the project.

2. Open the benchmark project

  Enter the directory `<path_to_tnn>/benchmark/benchmark_ios` and open the benchmark project.

3. Set developer account

  Click the benchmark project as shown below, find the project setting `Signing & Capabilities`, click the Team tab, and select` Add an Account ...`

  <div align=left ><img src="https://github.com/darrenyao87/tnn-models/raw/master/doc/cn/development/resource/ios_add_account_benchmark.jpg"/>

  Enter the Apple ID account and password in the following interface. After the addition is complete, return to the `Signing & Capabilities` interface and select the added account in the Team tab. If you don’t have an Apple ID, you can also use the `Create Apple ID` option to apply.

  `PS: There is no fee to apply for Apple ID, it can be passed immediately, and the APP can be debugged on the real machine after passing.`

  <div align=left ><img src="https://github.com/darrenyao87/tnn-models/raw/master/doc/cn/development/resource/ios_set_account.jpg"/>

4. Run on real machines

   4.1 Modify `Bundle Identifier`

   As shown in the figure, after the existing `Bundle Identifier`, a suffix (limited to numbers and letters) is randomly added to prevent personal accounts from encountering signature conflicts.

   <div align=left> <img src="https://github.com/darrenyao87/tnn-models/raw/master/doc/cn/development/resource/ios_set_bundleid_benchmark.jpg"/>

   4.2 Verify authorization

   For the first time, use the shortcut key `Command + Shift + K` to clean up the project, and then execute the shortcut key` Command + R` to run. If it is the first time to log in with Apple ID, Xcode will pop up a box and report the following error. You need to verify the authorization on the iOS device according to the prompt. Generally speaking, the authorization path on the phone is: Settings-> General-> Profile and Device Management-> Apple Development Options-> Click Trust

   <div align=left ><img src="https://github.com/darrenyao87/tnn-models/raw/master/doc/cn/development/resource/ios_verify_certificate_benchmark.jpg" width = "50%" height = "50%"/>

   
   4.3 Result

   For the first run, use the shortcut key `Command + Shift + K` to clean up the project, and then execute the shortcut key` Command + R` to run. Click the Run button on the interface, the interface will display the CPU and GPU time consumption of all models in the model directory. The running result of the iPhone7 real machine is shown below.

   <div align=left ><img src="https://github.com/darrenyao87/tnn-models/raw/master/doc/cn/development/resource/ios_benchmark_result.jpg" width = "50%" height = "50%"/>

   
   PS:

   a) Due to the different acceleration principles of GPU and CPU, the GPU performance of a specific model is not necessarily higher than that of the CPU, depending on the specific model/model structure/ engineering implementation. Everyone is welcome to participate in the development of TNN and make progress together.

   b) If you encounter the error message: `Unable to install ...`, please delete the existing benchmark app on the real device and run the installation again.
   
   c) If the CodeSign error `Command CodeSign failed with nonzero exit code` is encountered when the real device is running, please refer to the issue 20` iOS Demo Operation Step Instructions`

### II. Test the time cost on Android / ArmLinux platform

### 1. Build environment 

#### 1.1 Compile
 
Please refer to [TNN Compile Document](../user/compile_en.md):Compile for Android, to check if the environment meets the requirements.

#### 1.2 Execute

* adb command configuration  
Download[Android  tool](https://developer.android.com/studio/releases/platform-tools)，export `platform-tool` directory to`$PATH`。  
PS: If the adb version is too old，the script might fail to work, current adb verison：29.0.5-5949299
```
export PATH=<path_to_android_sdk>/platform-tools:$PATH
```

### 2. Add models

Put the model tnnproto into the models folder `<path_to_tnn>/benchmark/benchmark-model`，for example， 
```
cd <path_to_tnn>/benchmark/benchmark-model
cp mobilenet_v1.tnnproto .
```

### 3. Modify the script

Append model name to `benchmark_model_list` in `benchmark_models.sh`, such as：
```
 benchmark_model_list=(
 #test.tnnproto \
 mobilenet_v1.tnnproto \    # model name to be tested
)
```

### 4. Execute the script

```
./benchmark_models.sh  [-32] [-c] [-b] [-f] [-d] <device-id> [-t] <CPU/GPU>
Parameters：
    -32   build 32-bit library, default is 64
    -c    clean and recompile
    -b    only build, no execute
    -f    print out the time for every layer in network, otherwise the average time of all layers
    -t    add <CPU/GPU> to indicate the platform to run.
    -bs   executing binaries directly via shell
```
P.S. If -t is not set, the programme would run on CPU and GPU by default, "-t HUAWEI_NPU" needs to be specified to obtain Huawei NPU benchmark. 
#### 4.1 Overall Network Performance Analysis：

Analyze the overall network time-consuming and execute multiple times to obtain average performance.
Execute the script：

```
./benchmark_models.sh -c
```

The result is shown in the figure and saved to `benchmark_models_result.txt`.

<div align=left ><img src="https://github.com/darrenyao87/tnn-models/raw/master/doc/cn/development/resource/android_profiling.jpg" />

#### 4.2 Layer-by-layer Performance Analysis：

The layer-by-layer performance analysis tool is designed to calculate the running time of each layer and locate operator performance bottleneck.
Execute script:
```
./benchmark_models.sh -c -f
```
P.S. Huawei NPU does not support layer by layer analysis.
The result is shown in the figure and saved to `benchmark_models_result.txt`：
<div align=left ><img src="https://github.com/darrenyao87/tnn-models/raw/master/doc/cn/development/resource/opencl_profiling.jpg"/>


### 5.Special Instructions 

* For OpenCL，the purpose of the layer-by-layer performance analysis is to analyze the distribution of the kernel's time-consuming. There is an extra cost in order to print the information of each layer, and only the kernel time has reference significance. If you want to see the overall actual performance, the overall network performance analysis is more accurate.

* Compared with executing binaries directly via shell, the foreground benchmark app gets closer performance with an actual Android app. Due to Android's scheduler tailors behavior, it can result in observable differences in performance. Therefore, the benchmark app is preferred for performance measurement.
