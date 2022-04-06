# Code Architecture 

[中文版本](../../cn/development/architecture.md)

## I. API Design 

Considering the maintenance and compatibility of the open-source library, all externally exposed interfaces are managed uniformly through the include directory. For introduction of specific API, please refer to [API document](../user/api_en.md) 

## II. Model Reinterpreter 

The interface related to the model interpreter is abstracted, which can support multiple model formats' parsing. See the source/tnn/interpreter module for related codes. 

<div align=left ><img src="https://github.com/darrenyao87/tnn-models/raw/master/doc/cn/imgs/model_reinterpreter.png"/> 

AbstractModelInterpreter defines an abstract Interpret interface, and different model parsers parse different types of models. The interface related to DefaultModelInterpreter stores the relevant results in the NetStruture and NetResource structures. Some third-party models which cannot complete the interpretation would need a separate path such as CoreMLModelInterpreter, to complete third-party library adaptation. 

Different model parsers have their corresponding creators. 

```cpp 
// @brief ModelInterpreterCreator define model interpreter creator interface 
class ModelInterpreterCreator { 
public: 
virtual ~ ModelInterpreterCreator () {}; virtual AbstractModelInterpreter * CreateModelInterpreter () = 0;}; 

// @brief TypeModelInterpreterCreator create different type model interpreter 
template <typename T> 
class TypeModelInterpreterCreator: public ModelInterpreterCreator { 
virtual AbstractModelInterpreter * CreateModelInterpreter () { return new T (); }}; 
``` 

Different model interpreter creators are registered through Register. 

```cpp 
// @ brief TypeModelInterpreterRegister register TypeModelInterpreterCreator 
template <typename T> 
class TypeModelInterpreterRegister { 
public: 
TypeModelInterpreterRegister (ModelType type) { GetGlobalModelInterpreterCreatorMap () [type] = std :: shared_ptr <T> (new T ()); }}; 

``` 

Take the TNN model interpretation and registration as an example: TypeModelInterpreterRegister\<TypeModelInterpreterCreator\<ModelInterpreter>>g\_tnn\_model\_interpreter\_register(MODEL\_TYPE\_TNN); 

Through the TypeModelInterpreterRegister constructor, the TypeModelInterpreterCreator\<ModelInterpreter>corresponding to the TNN can be registered in the global model interpreter creator map, and then the corresponding creator can be obtained through the model type and the corresponding model interpreter can be constructed. 


## III. Network Construction 

The network construction mainly includes two parts, the first part is the network layer construction, and the second part is the blob node construction. 


```cpp 

// @ brief BaseLaye define the layer interface 
class BaseLayer { 
public: 
... 
virtual Status Init (Context * context, LayerParam * param, 
LayerResource * resource, std :: vector <Blob *> & inputs, std :: vector <Blob *> & outputs, AbstractDevice * device); 
...}; 

``` 

Similar to the previous model registration mechanism, different Layers will register different Layer Creators. After obtaining the corresponding Layer Creator through the Layer Type, the corresponding Layer can then be constructed. The corresponding output blob size can be calculated and the operator of the platform can be created afterward. 

The core of Blob node construction is memory allocation and optimization, which is mainly divided into blob memory recycling, blob memory splicing, and monitoring. 

<div align=left><img src="https://github.com/darrenyao87/tnn-models/raw/master/doc/cn/imgs/blob_memory.png"/> 

First of all, the memory between the blobs output by different layers will be cyclically reused through an internal algorithm. The memory reuse between different blobs will preferentially select blobs of similar size. 

After determining the blob memory reuse dependencies, the blob memory will be spliced and the memory will be allocated uniformly. Eventually, different blobs in the same Instance will have the same base pointer and different offsets. 
The memory between multiple instances of the same thread/different threads has the basis of memory reuse TNN internally provides a mechanism for automatic memory reuse between different instances within a single thread. Instances built through SHARE\_MEMORY\_MODE\_SHARE\_ONE\_THREAD will automatically implement multiple Instance memory reuse. At the same time, the Instance built by SHARE\_MEMORY\_MODE\_SET\_FROM\_EXTERNAL supports importing from external memory. The caller maintains the memory reuse relationship and memory allocation and release. For multi-thread reuse, it also needs to deal with inter-thread locking mechanism. 

## IV. Multi-platform Acceleration Operator Implementation 

<div align=left><img src="https://github.com/darrenyao87/tnn-models/raw/master/doc/cn/imgs/device.png"/> 

Abstract AbstractDevice interface, used to hide the implementation details of different Devices. Provide an interface for Device Memory size calculation, Device Memory allocation/release, CPU Memory and Device memory copy, Device Layer accelerated operator construction, and instance corresponding Device Context construction. 

```cpp 

// @brief AbstractDevice define create memory, context and layer acc interface. 
class AbstractDevice { 
public: 
... virtual BlobMemorySizeInfo Calculate (BlobDesc & desc) = 0; ... virtual Status Allocate (void ** handle, MatType mat_type, DimsVector dims) = 0; ... virtual Status Allocate (void ** handle, BlobMemorySizeInfo & size_info) = 0; ... virtual Status Free (void * handle) = 0; ... virtual Status CopyToDevice (BlobHandle * dst, const BlobHandle * src, BlobDesc & desc, void * command_queue) = 0; ... virtual Status CopyFromDevice (BlobHandle * dst, const BlobHandle * src, BlobDesc & desc, void * command_queue) = 0; ... virtual AbstractLayerAcc * CreateLayerAcc (LayerType type) = 0; ... virtual Context * CreateContext (int device_id) = 0; ...}; 
``` 


The network construction can obtain the corresponding Device implementation according to the configured DeviceType. Different Layers can build specific platform acceleration operators through the CreateLayerAcc interface, and interact through the unified abstract base class interface AbstractLayerAcc. 

```cpp 

// @brief AbstractLayerAcc define the layer acc interface 
class AbstractLayerAcc { 
public: 

... virtual Status Init (Context * context, LayerParam * param, 
LayerResource * resource, const std :: vector <Blob *> & inputs, const std :: vector <Blob *> & outputs) = 0; 
... virtual Status Forward (const std :: vector <Blob *> & inputs, 
const std :: vector <Blob *> & outputs) = 0;}; 

``` 

The same different LayerAcc is registered through the registration mechanism, and the Layer can construct different LayerAcc according to the LayerType. 

## V. Unit Testing 

TNN unit testing is based on googletest. Currently, unit testing is mainly built on Layer Acc and blob converter. The unit test uses the CPU Default implementation as the alignment benchmark to check the implementation of operators on different platforms. For specific unit test introduction, please refer to [unit test](./unit_test_en.md)