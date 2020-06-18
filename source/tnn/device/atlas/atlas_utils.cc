// Copyright 2019 Tencent. All Rights Reserved

#include "atlas_utils.h"
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>

namespace TNN_NS {

std::vector<std::string> SplitPath(const std::string& str, const std::set<char> delimiters) {
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

DataType ConvertFromAclDataType(aclDataType acl_datatype) {
    if (ACL_FLOAT == acl_datatype) {
        return DATA_TYPE_FLOAT;
    } else if (ACL_FLOAT16 == acl_datatype) {
        return DATA_TYPE_HALF;
    } else if (ACL_INT8 == acl_datatype) {
        return DATA_TYPE_INT8;
    } else if (ACL_INT32 == acl_datatype) {
        return DATA_TYPE_INT32;
    } else {
        return DATA_TYPE_FLOAT;
    }
}

DataFormat ConvertFromAclDataFormat(aclFormat acl_format) {
    if (ACL_FORMAT_NCHW == acl_format) {
        return DATA_FORMAT_NCHW;
    } else if (ACL_FORMAT_NHWC == acl_format) {
        return DATA_FORMAT_NHWC;
    } else {
        return DATA_FORMAT_AUTO;
    }
}

}  // namespace TNN_NS
