/*
*@file HiAiModelManagerService.h
*
* Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
*
*/
#ifndef __AI_MODEL_MANGER_SERVICE_H__
#define __AI_MODEL_MANGER_SERVICE_H__

#include <string>
#include <vector>
#include <map>

#include <mutex>

using namespace std;

namespace hiai {
using AIStatus = int32_t;

/* Error Numbers */
static const AIStatus AI_SUCCESS = 0;
static const AIStatus AI_FAILED = 1;
static const AIStatus AI_NOT_INIT = 2;
static const AIStatus AI_INVALID_PARA = 3;
static const AIStatus AI_TIMEOUT = 4;
static const AIStatus AI_ION_FAILED = 5;
static const AIStatus AI_TEE_FAILED = 6;
static const AIStatus AI_INVALID_API = 7;

enum AiModelDescription_Frequency {
    AiModelDescription_Frequency_UNSET = 0,
    AiModelDescription_Frequency_LOW = 1,
    AiModelDescription_Frequency_MEDIUM = 2,
    AiModelDescription_Frequency_HIGH = 3,
    AiModelDescription_Frequency_NONE = 255,
};

enum AiModelDescription_DeviceType {
    AiModelDescription_DeviceType_NPU = 0,
    AiModelDescription_DeviceType_IPU = 1,
    AiModelDescription_DeviceType_MLU = 2,
    AiModelDescription_DeviceType_CPU = 3,
    AiModelDescription_DeviceType_NONE = 255,
};

enum AiModelDescription_Framework {
    HIAI_FRAMEWORK_NONE = 0,
    HIAI_FRAMEWORK_TENSORFLOW,
    HIAI_FRAMEWORK_KALDI,
    HIAI_FRAMEWORK_CAFFE,
    HIAI_FRAMEWORK_TENSORFLOW_8BIT,
    HIAI_FRAMEWORK_CAFFE_8BIT,
    HIAI_FRAMEWORK_INVALID = 255,
};

enum AiModelDescription_ModelType{
    HIAI_MODELTYPE_ONLINE = 0,
    HIAI_MODELTYPE_OFFLINE
};

enum AiTensorImage_Format{
    AiTensorImage_YUV420SP_U8 = 0,
    AiTensorImage_XRGB8888_U8,
    AiTensorImage_YUV400_U8,

    AiTensorImage_INVALID = 255,
};

class AiContext {
public:
    string GetPara(const string &key) const;
    void AddPara(string &key, string &value);
	AIStatus GetAllKeys(vector<string> &keys);
private:
    map<string, string>         paras_;
};

class AiTensorDescription {
public:
    string GetPara(const string &key) const;
    void AddPara(string &key, string &value);
    AIStatus GetAllKeys(vector<string> &keys);
private:
    map<string, string>        paras_;
};

class TensorDimension {
public:
    TensorDimension();
    virtual ~TensorDimension();

    /*
    * @brief Tensor初始化接口
    * @param [in] number 输入的Tensor的数量
    * @param [in] channel 输入的Tensor的通道数
    * @param [in] height 输入的Tensor的高度
    * @param [in] width 输入的Tensor的宽度
    * @return 无
    */
    TensorDimension(uint32_t number, uint32_t channel, uint32_t height, uint32_t weight);

    void SetNumber(const uint32_t number);
    uint32_t GetNumber() const;
    void SetChannel(const uint32_t channel);
    uint32_t GetChannel() const;
    void SetHeight(const uint32_t height);
    uint32_t GetHeight() const;
    void SetWidth(const uint32_t width);
    uint32_t GetWidth() const;

private:
    uint32_t n{0};
    uint32_t c{0};
    uint32_t h{0};
    uint32_t w{0};
};

class AiModelMngerClient;
class AiTensor {
public:
    AiTensor();
    virtual ~AiTensor();

    /*
    * @brief Tensor初始化接口
    * @param [in] dim 输入tensor的尺寸结构信息
    * @return AIStatus::AI_SUCCESS 成功
    * @return Others 失败
    */
    AIStatus Init(const TensorDimension* dim);

    /*
    * @brief Tensor初始化接口
    * @param [in] number 输入的Tensor的数量
    * @param [in] height 输入的Tensor的高度
    * @param [in] width 输入的Tensor的宽度
    * @param [in] format 输入图片的格式AiTensorImage_Format类型
    * @return AIStatus::AI_SUCCESS 成功
    * @return Others 失败
    */
    AIStatus Init(uint32_t number, uint32_t height, uint32_t width, AiTensorImage_Format format);

    /*
    * @brief 获取Tensor buffer地址接口
    * @return void* tensor buffer地址
    */
    void *GetBuffer() const;

    /*
    * @brief 获取Tensor buffer内存大小
    * @return void* tensor buffer内存大小
    */
    uint32_t GetSize() const;

private:
    friend class AiModelMngerClient;
    void* buffer_{nullptr};
    uint32_t size_{0};
    TensorDimension tensor_dimension_;
    AiTensorDescription *tensor_desc_{nullptr};
};

class AiModelDescription {
public:
    /*
    * @brief AiModelDescription初始化描述的构造函数
    * @param [in] model_name 模型名称
    * @param [in] frequency 算力要求等级：AiModelDescription_Frequency
    * @param [in] framework 模型平台类型：AiModelDescription_Framework
    * @param [in] model_type 模型类型：AiModelDescription_ModelType
    * @param [in] device_type 设备类型：AiModelDescription_DeviceType
    */
    AiModelDescription(const string &model_name, const int32_t frequency,
                    const int32_t framework, const int32_t model_type, const int32_t device_type);
    virtual ~AiModelDescription();

    string GetName() const;
    void* GetModelBuffer() const;
    AIStatus SetModelBuffer(const void* data, uint32_t size);
    int32_t GetFrequency() const;
    int32_t GetFramework() const;
    int32_t GetModelType() const;
	int32_t GetDeviceType() const;
    uint32_t GetModelNetSize() const;

private:
    friend class AiModelMngerClient;
    string model_name_;
    int32_t frequency_{0};
    int32_t framework_{0};
	int32_t model_type_{0};
	int32_t device_type_{0};
    void *model_net_buffer_{nullptr};
    uint32_t model_net_size_{0};
    string model_net_key_;
};

class AiModelManagerClientListener {
public:
    virtual ~AiModelManagerClientListener(){}

    virtual void OnProcessDone(const AiContext &context, int32_t result, const vector<shared_ptr<AiTensor>> &out_tensor, int32_t iStamp) = 0;
    virtual void OnServiceDied() = 0;
};

class AiModelBuilder;
class MemBuffer {
public:
    /*
    * @brief 获取通用MEMBuffer的内存地址
    * @return MEMBuffer的内存地址
    */
    void* GetMemBufferData();

    /*
    * @brief 获取通用MEMBuffer的内存大小
    * @return MEMBuffer的内存大小
    */
    uint32_t GetMemBufferSize();

public:
    void *data_{nullptr};
private:
    friend class AiModelBuilder;
    void SetMemBufferSize(uint32_t size);
    void SetMemBufferData(void* data);
    uint32_t size_{0};
    void *servermem_{nullptr};
    bool isAppAlloc_{0};
};

class AiModelBuilder {
public:
    AiModelBuilder(shared_ptr<AiModelMngerClient> client = nullptr);

    virtual ~AiModelBuilder();

    /*
    * @brief OM离线模型在线编译接口
    * @param [in] input_membuffer 输入的OM离线模型buffer
    * @param [in] output_model_buffer 输出模型buffer
    * @param [out] output_model_size 输出模型大小
    * @return AIStatus::AI_SUCCESS 成功
    * @return Others 失败
    */
    AIStatus BuildModel(const vector<MemBuffer *> &input_membuffer, MemBuffer *output_model_buffer, uint32_t &output_model_size);

    /*
    * @brief 从文件读取OM离线模型proto信息
    * @param [in] path, 模型文件路径
    * @return MemBuffer * proto信息存储地址
    * @return nullptr 获取失败
    */
    MemBuffer* ReadBinaryProto(const string path);

    /*
    * @brief 从内存读取OM离线模型proto信息
    * @param [in] data, OM离线模型内存地址
    * @param [in] size, OM离线模型内存存储大小
    * @return MemBuffer * proto信息存储地址
    * @return nullptr 获取失败
    */
    MemBuffer* ReadBinaryProto(void* data, uint32_t size);

    /*
    * @brief 为输入OM离线模型用户内存buffer创建通用MemBuffer
    * @param [in] data, 模型用户内存地址
    * @param [in] size, 模型内存存储大小
    * @return MemBuffer * proto信息存储地址
    * @return nullptr 获取失败
    */
    MemBuffer* InputMemBufferCreate(void *data, uint32_t size);

    /*
    * @brief 为输入OM离线模型从文件创建MemBuffer
    * @param [in] path 文件路径
    * @return MemBuffer * 创建的输入MemBuffer内存指针
    * @return nullptr 创建失败
    */
    MemBuffer* InputMemBufferCreate(const string path);

    /*
    * @brief 为在线编译输出模型创建MemBuffer
    * @param [in] framework 模型平台类型
    * @param [in] input_membuffer 输入的OM离线模型buffer
    * @return MemBuffer * 创建的输出模型MemBuffer内存指针
    * @return nullptr 创建失败
    */
    MemBuffer* OutputMemBufferCreate(const int32_t framework, const vector<MemBuffer *> &input_membuffer);
    /*
    * @brief 注销MemBuffer内存，通过上述MemBuffer申请的内存最终都需要由此接口进行释放
    * @param [in] membuf, 创建的MemBuffer内存
    * @return void
    */
    void MemBufferDestroy(MemBuffer *membuf);

    /*
    * @brief 将模型buffer导出到文件
    * @param [in] membuf, 存储离线模型信息内存指针
    * @param [in] build_size, 离线模型大小
    * @param [in] build_path, 离线模型导出文件存储路径
    * @return AIStatus::AI_SUCCESS 导出成功
    * @return Others 导出失败
    */
    AIStatus MemBufferExportFile(MemBuffer *membuf, const uint32_t build_size, const string build_path);

private:
    shared_ptr<AiModelMngerClient> client_;
};

class AiModelMngerClient {
public:
    AiModelMngerClient();
    virtual ~AiModelMngerClient();

    /*
    * @brief 初始化接口
    * @param [in] listener 监听接口, 设置为nullptr时为同步调用, 否则为异步
    * @return AIStatus::AI_SUCCESS 成功
    * @return Others 失败
    */
    AIStatus Init(shared_ptr<AiModelManagerClientListener> listener);

    /*
    * @brief 加载模型
    * @param [in] model_desc 模型信息
    * @param [in] iStamp 异步返回标识，基于该标识和模型名称做回调索引
    * @return AIStatus::AI_SUCCESS 成功
    * @return AIStatus::AI_INVALID_API 失败，表示设备不支持NPU
    * @return Others 失败
    */
    AIStatus Load(vector<shared_ptr<AiModelDescription>> &model_desc);

    /*
    * @brief 模型处理接口, 运行加载模型的模型推理
    * @param [in] context, 模型运行上下文, 必须带model_name字段
    * @param [in] input_tensor, 模型输入节点tensor信息
    * @param [in] output_tensor, 模型输出节点tensor信息
    * @param [in] timeout, 推理超时时间
    * @param [in] iStamp 异步返回标识，基于该标识和模型名称做回调索引
    * @return AIStatus::AI_SUCCESS 成功
    * @return Others 失败
    */
    AIStatus Process(AiContext &context, vector<shared_ptr<AiTensor>> &input_tensor, vector<shared_ptr<AiTensor>> &output_tensor, uint32_t timeout, int32_t &iStamp);

    /*
    * @brief 模型兼容性检查接口
    * @param [in] model_desc, 模型描述
    * @param [out] isModelCompatibility, 兼容性检查标识
    * @return AIStatus::AI_SUCCESS 兼容性检查通过
    * @return Others 兼容性检查失败
    */
    AIStatus CheckModelCompatibility(AiModelDescription &model_desc, bool &isModelCompatibility);

    /*
    * @brief 获取模型输入输出tensor信息
    * @param [in] model_name, 模型名
    * @param [out] input_tensor 输出参数, 存储模型输入节点tensor信息
    * @param [out] output_tensor 输出参数, 存储模型输出节点tensor信息
    * @return AIStatus::AI_SUCCESS 获取成功
    * @return Others 获取失败
    */
    AIStatus GetModelIOTensorDim(const string& model_name, vector<TensorDimension>& input_tensor, vector<TensorDimension>& output_tensor);

    /*
    * @brief 获取DDK版本
    * @return char * DDK版本
    * @return nullptr 获取失败
    */
    char * GetVersion();

    /*
    * @brief 卸载模型
    * @return AIStatus::AI_SUCCESS 卸载成功
    * @return Others 卸载失败
    */
    AIStatus UnLoadModel();

private:
    static void onLoadDone(void* userdata, int taskStamp);
    static void onRunDone(void* userdata, int taskStamp);
    static void onUnloadDone(void* userdata, int taskStamp);
    static void onTimeout(void* userdata, int taskStamp);
    static void onError(void* userdata, int taskStamp, int errCode);
    static void onServiceDied(void* userdata);
    void SetModelManager(void* );
    void *GetModelManager() const;
    bool openclient();

private:
    friend class AiModelBuilder; 
    void *modelManager_{nullptr};
    void *listener_{nullptr};
    shared_ptr<AiModelManagerClientListener> cb_listener_;
    std::mutex syncRunMutex_;
    std::condition_variable condition_;
};

} //end namespace hiai

#endif
