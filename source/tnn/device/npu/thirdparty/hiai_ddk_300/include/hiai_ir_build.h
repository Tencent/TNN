#ifndef _H_HIAI_IR_BUILD_H_
#define _H_HIAI_IR_BUILD_H_
#include "graph/model.h"
namespace domi{

struct ModelBufferData
 {
	void* data;
	uint32_t length;
 };

class HiaiIrBuild{
public:


	bool CreateModelBuff(ge::Model& irModel,ModelBufferData& output);
	/**
	* @ingroup domi_omg
	* @brief 在线编译
	* @param [in] irModel 输入模型数据
	* @param [out] output 输出离线模型
	* @return Status 执行结果
	*/
	bool BuildIRModel(ge::Model& irModel,ModelBufferData& output);

	void ReleaseModelBuff(ModelBufferData& output);

};
}// namespace domi
#endif// _H_HIAI_IR_BUILD_H_

