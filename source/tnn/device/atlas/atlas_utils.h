// Copyright 2019 Tencent. All Rights Reserved

#ifndef TNN_SOURCE_DEVICE_ATLAS_ATLAS_UTILS_H_
#define TNN_SOURCE_DEVICE_ATLAS_ATLAS_UTILS_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "tnn/core/macro.h"
#include "hiaiengine/api.h"

namespace TNN_NS {

#define FORWARD_TIMEOUT 5000  // ms

#define USE_DEFINE_ERROR 0x6001
enum { HIAI_IDE_ERROR_CODE, HIAI_IDE_INFO_CODE, HIAI_IDE_WARNING_CODE };

HIAI_DEF_ERROR_CODE(USE_DEFINE_ERROR, HIAI_ERROR, HIAI_IDE_ERROR,
                    "user defined error");
HIAI_DEF_ERROR_CODE(USE_DEFINE_ERROR, HIAI_INFO, HIAI_IDE_INFO,
                    "user defined info");
HIAI_DEF_ERROR_CODE(USE_DEFINE_ERROR, HIAI_WARNING, HIAI_IDE_WARNING,
                    "user defined warning");

std::unordered_map<std::string, std::string> Kvmap(
    const hiai::AIConfig& config);

std::vector<std::string> SplitPath(const std::string& str,
                                   const std::set<char> delimiters);

long GetCurentTime();

int SaveMemToFile(std::string file_name, void* data, int size);

}  // namespace TNN_NS

#endif  // TNN_SOURCE_DEVICE_ATLAS_ATLAS_UTILS_H_
