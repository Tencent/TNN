// Copyright 2019 Tencent. All Rights Reserved

#ifndef TNN_SOURCE_TNN_INTERPRETER_RAPIDNETV3_ENCRYPTION_H_
#define TNN_SOURCE_TNN_INTERPRETER_RAPIDNETV3_ENCRYPTION_H_
#include <string>

namespace rapidnetv3 {
std::string BlurMix(const char *data , int len, bool encode);
void BlurMix(const char *data, char *dst, int len);
}  // namespace rapidnetv3



#endif  // TNN_SOURCE_TNN_INTERPRETER_RAPIDNETV3_ENCRYPTION_H_
