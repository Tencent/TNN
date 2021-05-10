// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "tnn/core/status.h"
#include "tnn/utils/string_format.h"

#include <iomanip>
#include <sstream>

namespace TNN_NS {

std::string StatusGetDefaultMessage(int code) {
    switch (code) {
        case TNNERR_INVALID_NETCFG:
            return "invalid net config, proto or model is invalid";
        case TNNERR_SET_CPU_AFFINITY:
            return "failed to set cpu affinity";
        case TNNERR_DEVICE_NOT_SUPPORT:
            return "device is nil or unsupported";
        case TNNERR_DEVICE_CONTEXT_CREATE:
            return "context is nil or created failed";
        default:
            return "";
    }
}

Status::~Status() {
    code_    = 0;
    message_ = "";
}

//constructor with code and message
Status::Status(int code, std::string message) {
    code_    = code;
    message_ = (message != "OK" && message.length() > 0) ? message : StatusGetDefaultMessage(code);
}

//int and status convert,assign,compare operator
Status& Status::operator=(int code) {
    code_    = code;
    message_ = StatusGetDefaultMessage(code);
    return *this;
}

bool Status::operator==(int code) {
    return code_ == code;
}

bool Status::operator!=(int code) {
    return code_ != code;
}

Status::operator int() {
    return code_;
}

//status convert to bool operator
Status::operator bool() {
    return code_ == TNN_OK;
}

//description with code(0x) and msg
std::string Status::description() {
    std::ostringstream os;
    os << "code: 0x" << std::uppercase << std::setfill('0') << std::setw(4) << std::hex << code_
       << " msg: " << message_;
    return os.str();
}

}  // namespace TNN_NS
