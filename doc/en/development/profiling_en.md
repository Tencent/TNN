# Model performance analysis
Analyze the time cost of a model

## I. Time cost test on the iOS platform
### Test steps
1. Add a test model

  Add test models in the `<path_to_tnn>/model` directory, each model has a folder, and the folder contains model files ending with proto and model. There is already a model squeezenetv1.1 in the project.

2. Open the benchmark project

  Enter the directory `<path_to_tnn>/benchmark/benchmark_ios` and open the benchmark project

3. Set developer account

  Click the benchmark project as shown below, find the project setting `Signing & Capabilities`, click the Team tab and select` Add an Account ...`

  <div align=left ><img src="https://raw.githubusercontent.com/darrenyao87/tnn-models/master/doc/cn/development/resource//ios_add_account_benchmark.jpg"/>

  Enter the Apple ID account and password in the following interface. After the addition is complete, return to the `Signing & Capabilities` interface and select the added account in the Team tab. If you don’t have an Apple ID, you can also use the `Create Apple ID` option to apply according to the relevant prompts.

  `PS: There is no fee to apply for Apple ID, it can be passed immediately, and the APP can be debugged on the real machine after passing.`

  <div align=left ><img src="https://raw.githubusercontent.com/darrenyao87/tnn-models/master/doc/cn/development/resource//ios_set_account.jpg"/>

4. Run on real devices

   4.1 Modify `Bundle Identifier`

   As shown in the figure, after the existing `Bundle Identifier`, a suffix (limited to numbers and letters) is randomly added to prevent personal accounts from encountering signature conflicts.

   <div align=left> <img src="https://raw.githubusercontent.com/darrenyao87/tnn-models/master/doc/cn/development/resource//ios_set_bundleid_benchmark.jpg"/>

   4.2 Verify authorization

   For the first time, use the shortcut key `Command + Shift + K` to clean up the project, and then execute the shortcut key` Command + R` to run. If it is the first time to log in with Apple ID, Xcode will pop up a box and report the following error. You need to verify the authorization on the iOS device according to the prompt. Generally speaking, the authorization path on the phone is: Settings-> General-> Profile and Device Management-> Apple Development Options-> Click Trust

   <div align=left ><img src="https://raw.githubusercontent.com/darrenyao87/tnn-models/master/doc/cn/development/resource//ios_verify_certificate_benchmark.jpg" width = "50%" height = "50%"/>

   
   4.3 Result

   For first run, use the shortcut key `Command + Shift + K` to clean up the project, and then execute the shortcut key` Command + R` to run. Click the Run button on the interface, the interface will display the CPU and GPU time consumption of all models in the model directory. The running result of the iPhone7 real machine is shown below.

   <div align=left ><img src="https://raw.githubusercontent.com/darrenyao87/tnn-models/master/doc/cn/development/resource//ios_benchmark_result.jpg" width = "25%" height = "25%"/>

   
   PS:

   a) Due to the different acceleration principles of GPU and CPU, the GPU performance of a specific model is not necessarily higher than that of the CPU, which is related to the specific model, model structure and engineering implementation. Everyone is welcome to participate in the development of TNN and make progress together.

   b) If you encounter the error message: `Unable to install ...`, please delete the existing benchmark app on the real device and run the installation again.
   
   c) If the CodeSign error `Command CodeSign failed with nonzero exit code` is encountered when the real device is running, please refer to the issue` iOS Demo Operation Step Instructions`
   https://git.code.oa.com/deep_learning_framework/TNN/issues/20

### II. Android / ArmLinux platform time-consuming test
### 1. Build Environment
#### 1.1 Compilation environment  
Please refer to [TNN Compile Document](../user/compile_en.md):Compile for Android, to check if the environment meets the requirements.

#### 1.2 Execution environment
* adb command configuration  
Download[Android  tool](https://developer.android.com/studio/releases/platform-tools)，export `platform-tool` directory to`$PATH`。  
PS: If the adb version is too old，the scipr might fail to work, current adb verison：29.0.5-5949299
```
export PATH=<path_to_android_sdk>/platform-tools:$PATH
```

### 2. Add models
Make a new directory 'models' under `<path_to_tnn>/platforms/android`，and put the model prototxti nto models folder，for example， 
```
cd <path_to_tnn>/platforms/android
mkdir models
cp mobilenet_v1.onnx.opt.onnx.rapidproto models
```

### 3. Modify the script
Append model name to `benchmark_model_list` in `profiling_model_android.sh`, such as：
```
 benchmark_model_list=(
 #test.rapidproto \
 mobilenet_v1.onnx.opt.onnx.rapidproto \    # model name to be tested
)
```

### 4. Execute the script
```
./profiling_model_android.sh  <-c> <-64> <-p> <-b> <-f> <-d> <device-id>
Parameters：
    -c    clean and recompile
    -64   use 64-bit library, default is 32
    -p    replace the models
    -b    only build
    -f    print out the time for every layer in network, otherwise the average time of all layers
    -d    add <device-id> to indicate the programme runs on which deivce when multiple devices connected
```
#### 4.1 Overall Network Performance Analysis：

Analyze the overall network time-consuming and execute multiple times to obtain average performance.
Execute the script：
```
./profiling_model_android.sh -c -64 -p
```
The result is shown in the figure and is saved to `dump_data/test_log.txt`.

<div align=left ><img src="https://raw.githubusercontent.com/darrenyao87/tnn-models/master/doc/cn/development/resource/android_profiling.jpg" />

#### 4.2 Layer-by-layer Performance Analysis：
The layer-by-layer performance analysis tool is designed to calculate the running time of each layer and locate operator performance bottleneck.
Execute script:
```
./profiling_model_android.sh -c -64 -p -f
```
The result is shown in the figure and saved to `dump_data/test_log.txt`：
<div align=left ><img src="https://raw.githubusercontent.com/darrenyao87/tnn-models/master/doc/cn/development/resource/opencl_profiling.jpg"/>  


### 5.Special Instructions 
* For OpenCL，the purpose of the layer-by-layer performance analysis is to analyze the distribution of the kernel's time-consuming. There is an extra cost in order to print the information of each layer, and only the kernel time has reference significance. If you want to see the overall actual performance, the overall network performance analysis is more accurate.