# 模型性能分析

[English Version](../../en/development/profiling_en.md)

分析模型耗时情况

## 一、iOS平台耗时测试
### 测试步骤
1. 添加测试模型

   在`<path_to_tnn>/model`目录下添加测试模型，每个模型一个文件夹，文件夹中包含以proto和model结尾的模型文件。目前工程中已有模型squeezenetv1.1

2. 打开benchmark工程

   进入目录`<path_to_tnn>/benchmark/benchmark_ios`，双击打开benchmark工程

3. 设置开发者账号

   如下图点击benchmark工程，找到工程设置`Signing & Capabilities`，点击Team选项卡选择`Add an Account...`

   <div align=left ><img src="https://github.com/darrenyao87/tnn-models/raw/master/doc/cn/development/resource/ios_add_account_benchmark.jpg" width = "75%" height = "75%"/>

   在如下界面输入Apple ID账号和密码，添加完成后回到`Signing & Capabilities`界面，并在Team选项卡中选中添加的账号。如果没有Apple ID也可以通过`Create Apple ID`选项根据相关提示进行申请。

    `PS：申请Apple ID无需付费，可以即时通过，通过后才可在真机上运行APP调试`

   <div align=left ><img src="https://github.com/darrenyao87/tnn-models/raw/master/doc/cn/development/resource/ios_set_account.jpg" width = "75%" height = "75%"/>


4. 真机运行  

   4.1 修改`Bundle Identitifier`

   如图在现有`Bundle Identifier`后随机添加后缀（限数字和字母），避免个人账户遇到签名冲突。

   <div align=left ><img src="https://github.com/darrenyao87/tnn-models/raw/master/doc/cn/development/resource/ios_set_bundleid_benchmark.jpg" width = "75%" height = "75%"/>

   4.2 验证授权

   首次运行先利用快捷键`Command + Shift + K`对工程进行清理，再执行快捷键`Command + R`运行。如果是首次登陆Apple ID，Xcode会弹框报如下错误，需要在iOS设备上根据提示进行授权验证。一般来说手机上的授权路径为：设置 -> 通用 -> 描述文件与设备管理 -> Apple Development选项 -> 点击信任

   <div align=left ><img src="https://github.com/darrenyao87/tnn-models/raw/master/doc/cn/development/resource/ios_verify_certificate_benchmark.jpg" width = "75%" height = "75%"/>

   4.3 运行结果

   首次运行先利用快捷键`Command + Shift + K`对工程进行清理，再执行快捷键`Command + R`运行。在界面上点击Run按钮，界面会显示model目录下所有模型的CPU和GPU耗时情况。iPhone7真机运行结果如下图。

   <div align=left ><img src="https://github.com/darrenyao87/tnn-models/raw/master/doc/cn/development/resource/ios_benchmark_result.jpg" width = "50%" height = "50%"/>

   PS：

   a) 由于GPU和CPU加速原理不同，具体模型的GPU性能不一定比CPU高，与具体机型、模型结构以及工程实现有关。欢迎大家参与到TNN开发中，共同进步。

   b) 如遇到`Unable to install...`错误提示，请在真机设备上删除已有的benchmark app，重新运行安装。

   c) 真机运行时，如果遇到CodeSign错误`Command CodeSign failed with a nonzero exit code`，可参看issue20 `iOS Demo运行步骤说明`

## 二、Android平台耗时测试
### 1. 环境搭建  
#### 1.1 编译环境  
参考[TNN编译文档](../user/compile.md) 中Android库编译，检查环境是否满足要求。  

#### 1.2 执行环境  
* adb命令配置  
下载[安卓SDK工具](https://developer.android.com/studio/releases/platform-tools)，将`platform-tool`目录加入`$PATH`环境变量中。  
PS: 如果adb版本过低，可能执行脚本会失败。当前测试的adb版本为：29.0.5-5949299
```
export PATH=<path_to_android_sdk>/platform-tools:$PATH
```

### 2. 添加模型
在`<path_to_tnn>/benchmark/benchmark-model`目录下，将要测试模型的tnnproto放入文件夹，例如，
```
cd <path_to_tnn>/benchmark/benchmark-model
cp mobilenet_v1.tnnproto .
```


### 3. 修改脚本
在脚本`benchmark_models.sh`中的`benchmark_model_list`变量里添加模型文件名，例如：
```
 benchmark_model_list=(
 #test.tnnproto \
 mobilenet_v1.tnnproto \    # 待测试的模型文件名
)
```

### 4. 执行脚本
```
./benchmark_models.sh  [-32] [-c] [-b] [-f] [-d] [-bs] <device-id> [-t] <CPU/GPU>
参数说明：
    -32   编译32位的库，否则为64位
    -c    删除之前的编译文件，重新编译
    -b    仅编译，不执行
    -f    打印每一层的耗时，否则是整个网络的平均耗时。
    -t    指定执行的平台。需要加上<CPU/GPU/HUAWEI_NPU>
    -bs   shell运行可执行文件测试
```
P.S. 不指定 -t, 默认跑CPU和GPU, 华为npu benchmark需通过-t HUAWEI_NPU特殊制定.
#### 4.1 全网络性能分析：
分析整体网络耗时，执行多次，获取平均性能。  
执行脚本：
```
./benchmark_models.sh -c
```
结果如图：
<div align=left ><img src="https://github.com/darrenyao87/tnn-models/raw/master/doc/cn/development/resource/android_profiling.jpg" width = "75%" height = "75%"/>

执行结果会保存在`benchmark_models_result.txt`中。


#### 4.2 逐层性能分析：
逐层性能分析工具可准备计算各层耗时，以便进行模型优化和op性能问题定位。  
执行脚本:
```
./benchmark_models.sh -c -f
```
结果如图：
<div align=left ><img src="https://github.com/darrenyao87/tnn-models/raw/master/doc/cn/development/resource/opencl_profiling.jpg" width = "75%" height = "75%"/>

执行结果会保存在`benchmark_models_result.txt`中。  
P.S. 华为npu不支持每层分析。

### 5. 特殊说明
* 对于OpenCL平台，逐层性能分析的目的是分析kernel的耗时分布，其中为了打印每层耗时，有额外开销，只有kernel时间具有参考意义。如果要看整体实际性能，需要参考全网络性能分析。
* Android系统相比shell执行可执行文件耗时测试，app耗时测试的性能更贴近真实安卓app执行的性能。受安卓调度策略的影响，两种方式的性能可能有明显差异。综上所述，安卓app耗时测试更为推荐。
