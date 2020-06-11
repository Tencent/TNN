#ifndef __HIAI_MIX_MODEL_API_H__
#define __HIAI_MIX_MODEL_API_H__

#include <cstdint>


/**
This is the HIAI MixModel Runtime C API:
*/

/**
* HIAIMixModel.h提供的API在HIAIModelManager.h提供的API基础上修改和新增，修改涉及三点
* 1 API 函数名称和自定义数据结果名称统一增加关键字"Mix"
* 2 部分API接口增加参数HIAI_MixFramework，以区别是分割模型还是非分割模型
* 3 HIAI_MixModel_LoadFromModelBuffers 接口每次只支持load 一个模型
*/

#ifdef __cplusplus
extern "C" {
#endif
typedef enum {
    MODEL_MANAGER_OK                  = 0,
    MODEL_NAME_LEN_ERROR              =  1,
    MODEL_DIR_ERROR                   =  2,
    MODEL_SECRET_KEY_ERROR            =  3,
    MODEL_PARA_SECRET_KEY_ERROR       =  4,
    FRAMEWORK_TYPE_ERROR              =  5,
    MODEL_TYPE_ERROR                  =  6,
    IPU_FREQUENCY_ERROR               =  7,
    MODEL_NUM_ERROR                   =  8,
    MODEL_SIZE_ERROR                  =  9,
    TIMEOUT_ERROR                     = 10,
    INPUT_DATA_SHAPE_ERROR            = 11,
    OUTPUT_DATA_SHAPE_ERROR           = 12,
    INPUT_DATA_NUM_ERROR              = 13,
    OUTPUT_DATA_NUM_ERROR             = 14,
    MODEL_MANAGER_TOO_MANY_ERROR      = 15,
    MDOEL_BUILD_TYPE_INCONSISTENT     = 16,
    MDOEL_FRAMEWORK_TYPE_INCONSISTENT = 17,
    MODEL_NAME_DUPLICATE_ERROR        = 18,

    HIAI_SERVER_CONNECT_ERROR         = 19,
    HIAI_SERVER_CONNECT_IRPT          = 20,

    DLOPEN_ERROR                      = 21,
    DLSYM_ERROR                       = 22,
    BUILD_MODEL_ERROR                 = 23,
    MODEL_MULTI_SEG_ERROR             = 24,

    //--------start for caffe/tf executor
    MODEL_CONTENT_ERROR               = 25, // return this error code when load xxx.caffemodel or xx.pb failed
    MODEL_PARAMETER_ERROR             = 26, // return this error code when parse xxx.prototxt or xx.param.txt failed
    NOT_SUPPORT_ERROR                 = 27, // return this error code when not support this function
    ALLOC_MEM_ERROR                   = 28,
    NULL_POINTER_ERROR                = 29,
    PARSE_MODEL_ERROR                 = 30,
    FILE_NOT_FOUND_IN_ZIP             = 31,
    ADD_FILE_ALREADY_EXIT_IN_ZIP      = 32,
    OPEN_NEW_FILE_IN_ZIP_ERROR        = 33,
    WRITE_FILE_IN_ZIP_ERROR           = 34,
    CLOSE_FILE_IN_ZIP_ERROR           = 35,
    CLOSE_ZIP_ERROR                   = 36,
    GET_ZIP_GLOBAL_INFO_ERROR         = 37,
    FILE_NUM_IN_ZIP_EXCEED_512        = 38,
    GET_CURRENT_FILE_INFO_ERROR       = 39,
    OPEN_CURRENT_FILE_ERROR           = 40,
    READ_CURRENT_FILE_ERROR           = 41,
    CLOSE_CURRENT_FILE_ERROR          = 42,
    GOTO_NEXT_FILE_ERROR              = 43,
    MEMBER_BUFFER_SIZE_ERROR          = 44,
    INPUT_FILE_NAME_ERROR             = 45,
    MANAGER_IS_INVALID                = 46,
    MODELBUFFER_IS_INVALID            = 47,
    PARA_IS_INVALID                   = 48,
    MODEL_HAS_NOT_BE_LOADED           = 49,

    //--------end for caffe/tf executor
    MODEL_TENSOR_SHAPE_NO_MATCH       = 500,

    EXPIRATION_FUCNTION               = 999,
    INTERNAL_ERROR                    = 1000,
} MIX_MODEL_MANAGER_ERRCODE;


/*对HIAI_Framework重命名，未修改*/
typedef enum {
    HIAI_MIX_FRAMEWORK_NONE            = 0,
    HIAI_MIX_FRAMEWORK_TENSORFLOW      = 1,
    HIAI_MIX_FRAMEWORK_KALDI           = 2,
    HIAI_MIX_FRAMEWORK_CAFFE           = 3,
    HIAI_MIX_FRAMEWORK_TENSORFLOW_8BIT = 4,
    HIAI_MIX_FRAMEWORK_CAFFE_8BIT      = 5,
    HIAI_MIX_FRAMEWORK_INVALID,
} HIAI_Mix_Framework;

/*对HIAI_DevPerf重命名，未修改*/
typedef enum {
	HIAI_MIX_DEVPERF_UNSET = 0,
	HIAI_MIX_DEVPREF_LOW,
	HIAI_MIX_DEVPREF_NORMAL,
	HIAI_MIX_DEVPREF_HIGH,
} HIAI_MixDevPerf;

/*对HIAI_DataType重命名，未修改*/
typedef enum {
	HIAI_MIX_DATATYPE_UINT8 = 0,
	HIAI_MIX_DATATYPE_FLOAT32 = 1,
} HIAI_MixDataType;

/*在HIAI_ModelManagerListener基础上，删除onServiceDied*/
typedef struct HIAI_MixModelListener_struct
{
    void(*onLoadDone)(void* userdata, int taskStamp);
    void(*onRunDone)(void* userdata, int taskStamp);
    void(*onUnloadDone)(void* userdata, int taskStamp);
    void(*onTimeout)(void* userdata, int taskStamp);
    void(*onError)(void* userdata, int taskStamp, int errCode);    
    void (*onServiceDied)(void* userdata);
    void* userdata;
} HIAI_MixModelListener;

/*对HIAI_ModelTensorInfo重命名，未修改*/
typedef struct
{
	int input_cnt;
	int output_cnt;
	int *input_shape;
	int *output_shape;
} HIAI_MixModelTensorInfo;

/*对HIAI_TensorDescription重命名，未修改*/
typedef struct {
	int number;
	int channel;
	int height;
	int width;
	HIAI_MixDataType dataType;  /* optional */
} HIAI_MixTensorDescription;

typedef	struct HIAI_MixModelManager HIAI_MixModelManager;
typedef  struct HIAI_MixModelBuffer HIAI_MixModelBuffer;
typedef  struct HIAI_MixTensorBuffer HIAI_MixTensorBuffer;
typedef  struct HIAI_MixMemBuffer  HIAI_MixMemBuffer;

/*-----------------------ModelManager-----------------------------------------------------------*/

HIAI_MixModelManager* HIAI_MixModelManager_Create(HIAI_MixModelListener* listener);
void HIAI_MixModelManager_Destroy(HIAI_MixModelManager* manager);

/*-----------------------ModelBuffer---------------------------------------------*/

/*modify :add HIAI_MixFramework type*/
HIAI_MixModelBuffer* HIAI_MixModelBuffer_Create_From_File(const char* name, const char* path, HIAI_MixDevPerf perf,bool mixflag);
HIAI_MixModelBuffer* HIAI_MixModelBuffer_Create_From_Buffer(const char* name, void* modelbuf, uint32_t size, HIAI_MixDevPerf perf,bool mixflag);
void HIAI_MixModelBuffer_Destroy(HIAI_MixModelBuffer* buffer);
const char* HIAI_MixModelBuffer_GetName(HIAI_MixModelBuffer* b);
const char* HIAI_MixModelBuffer_GetPath(HIAI_MixModelBuffer* b);
const void* HIAI_MixModelBuffer_GetData(HIAI_MixModelBuffer* b);
HIAI_MixDevPerf HIAI_MixModelBuffer_GetPerf(HIAI_MixModelBuffer* b);
int HIAI_MixModelBuffer_GetSize(HIAI_MixModelBuffer* b);

/*-----------------------TensorBuffer--------------------------------------------*/
HIAI_MixModelTensorInfo* HIAI_MixModel_GetModelTensorInfo(HIAI_MixModelManager* manager, const char* modelname);//加载后
void HIAI_MixModel_ReleaseModelTensorInfo(HIAI_MixModelTensorInfo* modeltensor);

HIAI_MixTensorBuffer* HIAI_MixTensorBuffer_Create(int n, int c, int h, int w);
void HIAI_MixTensorBufferr_Destroy(HIAI_MixTensorBuffer* buffer);
HIAI_MixTensorDescription HIAI_MixTensorBuffer_getTensorDesc(HIAI_MixTensorBuffer* b);
void*  HIAI_MixTensorBuffer_GetRawBuffer(HIAI_MixTensorBuffer* b);
int  HIAI_MixTensorBuffer_GetBufferSize(HIAI_MixTensorBuffer* b);

/*-----------------------MemBuffer---------------------------------------------*/
HIAI_MixMemBuffer* HIAI_MixMemBuffer_Create_From_File(const char *path);
HIAI_MixMemBuffer* HIAI_MixMemBuffer_Create_From_Buffer(void* buffer, const uint32_t size);
void HIAI_MixMemBuffer_Destroy(HIAI_MixMemBuffer* membuf);
void* HIAI_MixMemBuffer_GetData(HIAI_MixMemBuffer* b);
int HIAI_MixMemBuffer_GetSize(HIAI_MixMemBuffer* b);

/*-----------------------CheckModelCompatibility---------------------------------------------*/

/*增加frameworkType参数*/
bool HIAI_CheckMixModelCompatibility_From_File(HIAI_MixModelManager* manager,bool mixflag, const char *modelpath);
bool HIAI_CheckMixModelCompatibility_From_Buffer(HIAI_MixModelManager* manager, bool mixflag, void* buffer, const uint32_t size);

/*-----------------------build model---------------------------------------------*/

// 从文件接口进行在线编译
int HIAI_MixModel_BuildModel_FromPath(HIAI_MixModelManager* manager,
										HIAI_Mix_Framework frameworktype,
										const char* modelpath,
										const char* modelparapath,
										const char* offlinemodelpath,
										bool mixflag);

// 从内存接口进行在线编译
int HIAI_MixModel_BuildModel_FromBuffer(HIAI_MixModelManager* manager,
										HIAI_Mix_Framework frameworktype,
										const void* modelbuffer,
										unsigned int modelbuffersize,
										const void* modelparabuffer,
										unsigned int modelparabuffersize,
										const char* offlinemodelpath,
										bool mixflag);

//HIAI_MixModelBuffer 含size
int HIAI_MixModel_BuildModel_ToMem(HIAI_MixModelManager* manager,
										HIAI_Mix_Framework frameworktype,
										HIAI_MixMemBuffer* inputmodelbuffers[],
										unsigned int inputmodelbuffersnum,
										HIAI_MixMemBuffer* outputmodelbuffer,
										unsigned int *outmodelsize,
										bool mixflag);

/*-----------------------load model---------------------------------------------*/
int HIAI_MixModel_LoadFromModelBuffers(HIAI_MixModelManager* manager,HIAI_MixModelBuffer* bufferarray[], int nbuffers);


/*-----------------------run model---------------------------------------------*/
// timeout 单位毫秒
int HIAI_MixModel_RunModel(HIAI_MixModelManager* manager, HIAI_MixTensorBuffer* inputs[], uint32_t niput,
                                                        HIAI_MixTensorBuffer* outputs[], uint32_t noutput, uint32_t timeout, const char* modelname);


/*-----------------------unload model---------------------------------------------*/
int HIAI_MixModel_UnLoadModel(HIAI_MixModelManager* manager);


/*-----------------------other---------------------------------------------*/
const char* HIAI_MixModel_GetVersion();
const char* HIAI_ModelManager_GetVersion(HIAI_MixModelManager* manager);
void HIAI_MixModel_SetBufferMultiple(HIAI_MixModelManager* manager,uint32_t multiple);
const uint32_t HIAI_MixModel_GetBufferMultiple(HIAI_MixModelManager* manager);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // __HIAI_MIX_MODEL_API_H__

