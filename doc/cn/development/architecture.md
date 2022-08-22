# 架构详解

[English Version](../../en/development/architecture_en.md)

## 一、API设计
考虑开源库后期维护及版本兼容性，所有对外暴露接口均通过include目录统一管理。具体API相关介绍可参见[API文档](../user/api.md)


## 二、模型解析

对模型解析相关接口进行了抽象，可支持多种模型格式解析和扩充，相关代码见source/tnn/interpreter模块。

 <div align=left ><img src="https://github.com/darrenyao87/tnn-models/raw/master/doc/cn/imgs/model_reinterpreter.png"/>

AbstractModelInterpreter定义了抽象的Interpret接口，不同的模型解析器解析不同类型模型。DefaultModelInterpreter相关的接口将相关结果存入NetStruture和NetResource结构中，部分第三方模型无法完成内部结构解析的有单独适配，如CoreMLModelInterpreter，以完成第三方库适配。

不同的模型解析器均有对应的creator

```cpp
// @brief ModelInterpreterCreator define model interpreter creator interface
class ModelInterpreterCreator {
public:
    virtual ~ModelInterpreterCreator() {};
    virtual AbstractModelInterpreter* CreateModelInterpreter() = 0;
};

// @brief TypeModelInterpreterCreator create different type model interpreter
template<typename T>
class TypeModelInterpreterCreator:public ModelInterpreterCreator {
    virtual AbstractModelInterpreter* CreateModelInterpreter() {
        return new T();
    }
};
```

不同的model interpreter creator均通过Register进行注册。

```cpp
//@brief TypeModelInterpreterRegister register TypeModelInterpreterCreator
template <typename T>
class TypeModelInterpreterRegister {
public:
    TypeModelInterpreterRegister(ModelType type) {
        GetGlobalModelInterpreterCreatorMap()[type] = std::shared_ptr<T>(new T());
    }
};

```

以TNN模型解析注册为例： TypeModelInterpreterRegister\<TypeModelInterpreterCreator\<ModelInterpreter>> g\_tnn\_model\_interpreter\_register(MODEL\_TYPE\_TNN);

通过TypeModelInterpreterRegister构造函数，可将TNN对应的TypeModelInterpreterCreator\<ModelInterpreter>注册到全局model interpreter creator map中，后续通过model type即可获取对应creator并构建对应的model interpreter。


## 三、网络构建

网络构建主要包含两大部分，第一部分为网络Layer构建，第二部分为Blob节点构建。


```cpp

//@brief BaseLaye define the layer interface
class BaseLayer {
public:

    ...

    virtual Status Init(Context* context, LayerParam* param,
                        LayerResource* resource, std::vector<Blob*>& inputs,
                        std::vector<Blob*>& outputs,
                        AbstractDevice* device);

    ...
};

```

与前面模型注册机制类似，不同Layer会注册不同的Layer Creator。通过Layer Type获取对应的Layer Creator后即可构建出对应的Layer,Layer构建完成后可计算对应输出blob尺寸以及创建平台加速算子。

Blob节点构建核心在于内存的分配和优化，主要分为blob内存循环复用，blob内存拼接与监控。

<div align=left><img src="https://github.com/darrenyao87/tnn-models/raw/master/doc/cn/imgs/blob_memory.png"/>

首先不同layer输出blob间内存会通过内部算法实现循环复用，不同blob间内存复用会优先选择尺寸接近的blob。

确定blob内存复用关系后，会对blob内存进行拼接，并统一分配内存，最终同一Instance不同blob间持有相同的base指针以及不同的偏移量，同一线程多个instance间以及不同线程instance间内存有了内存复用的基础。TNN内部提供了单一线程内不同instance间内存复用自动实现机制，通过SHARE\_MEMORY\_MODE\_SHARE\_ONE\_THREAD构建的Instance会自动实现多Instance内存复用。同时SHARE\_MEMORY\_MODE\_SET\_FROM\_EXTERNAL构建的Instance支持内存外部传入，由调用者维护内存复用关系以及内存分配释放，对于多线程复用还需要处理线程间加锁机制。

## 四、多平台加速算子实现

<div align=left><img src="https://github.com/darrenyao87/tnn-models/raw/master/doc/cn/imgs/device.png"/>

抽象AbstractDevice接口，用于隐藏不同Device实现细节。提供Device Memory 尺寸计算，Device Memory分配释放，内存CPU Memory与Device meomoy拷贝，Device Layer加速算子构建，以及Instance对应Device Context构建等接口。

```cpp
// @brief AbstractDevice define create memory, context and layer acc interface.
class AbstractDevice {
public:
    ...
    virtual BlobMemorySizeInfo Calculate(BlobDesc& desc) = 0;
    ...
    virtual Status Allocate(void** handle, MatType mat_type, DimsVector dims) = 0;
    ...
    virtual Status Allocate(void** handle, BlobMemorySizeInfo& size_info) = 0;
    ...
    virtual Status Free(void* handle) = 0;
    ...
    virtual Status CopyToDevice(BlobHandle* dst, const BlobHandle* src,
                                BlobDesc& desc, void* command_queue) = 0;
    ...
    virtual Status CopyFromDevice(BlobHandle* dst, const BlobHandle* src,
                                  BlobDesc& desc, void* command_queue) = 0;
    ...
    virtual AbstractLayerAcc* CreateLayerAcc(LayerType type) = 0;
    ...
    virtual Context* CreateContext(int device_id) = 0;
    ...
};
```

网络构建根据配置的DeviceType可获取对应的Device实现，不同的Layer通过CreateLayerAcc接口即可构建特定平台加速算子，并通过统一的抽象基类接口AbstractLayerAcc进行交互。

```cpp

// @brief AbstractLayerAcc define the layer acc interface
class AbstractLayerAcc {
public:

    ...

    virtual Status Init(Context *context, LayerParam *param,
                        LayerResource *resource,
                        const std::vector<Blob *> &inputs,
                        const std::vector<Blob *> &outputs) = 0;

    ...

    virtual Status Forward(const std::vector<Blob *> &inputs,
                           const std::vector<Blob *> &outputs) = 0;
};

```

同样不同的LayerAcc通过注册机制进行注册，Layer根据LayerType即可构建不同的LayerAcc。

## 五、单元测试

TNN 单元测试基于googletest构建，当前主要对Layer Acc以及blob converter构建了单元测试。单元测试以CPU Default实现为对齐基准，以监控不同平台加速算子实现，具体单元测试相关介绍可参见[单元测试](unit_test.md)
