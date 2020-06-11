// Copyright 2019 Tencent. All Rights Reserved

#include "atlas_utils.h"
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>

namespace TNN_NS {

std::unordered_map<std::string, std::string> Kvmap(
    const hiai::AIConfig& config) {
    std::unordered_map<std::string, std::string> kv;
    for (int index = 0; index < config.items_size(); ++index) {
        const ::hiai::AIConfigItem& item = config.items(index);
        kv.insert(std::make_pair(item.name(), item.value()));
    }
    return std::move(kv);
}

std::vector<std::string> SplitPath(const std::string& str,
                                   const std::set<char> delimiters) {
    std::vector<std::string> result;
    char const* pch   = str.c_str();
    char const* start = pch;
    for (; *pch; ++pch) {
        if (delimiters.find(*pch) != delimiters.end()) {
            if (start != pch) {
                std::string str(start, pch);
                result.push_back(str);
            } else {
                result.push_back("");
            }
            start = pch + 1;
        }
    }
    result.push_back(start);
    return result;
}

long GetCurentTime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

int SaveMemToFile(std::string file_name, void* data, int size) {
    FILE* fd = fopen(file_name.c_str(), "wb");
    if (fd == nullptr) {
        return -1;
    }

    int ret = fwrite(data, 1, size, fd);
    if (ret != size) {
        fclose(fd);
        return -1;
    }

    fclose(fd);
    return 0;
}

}  // namespace TNN_NS
