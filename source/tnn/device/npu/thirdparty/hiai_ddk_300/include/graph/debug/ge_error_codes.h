/**
 * Copyright (C)  2019. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file status_utils.h
 *
 * @brief
 *
 * @version 1.0
 *
 */

#ifndef GE_ERROR_CODES_H
#define GE_ERROR_CODES_H

namespace ge {

#ifdef HOST_VISIBILITY
    #define GE_FUNC_HOST_VISIBILITY __attribute__((visibility("default")))
#else
    #define GE_FUNC_HOST_VISIBILITY
#endif
#ifdef DEV_VISIBILITY
    #define GE_FUNC_DEV_VISIBILITY __attribute__((visibility("default")))
#else
    #define GE_FUNC_DEV_VISIBILITY
#endif

using graphStatus = uint32_t;
const graphStatus GRAPH_FAILED = 0xFFFFFFFF;
const graphStatus GRAPH_SUCCESS = 0;
const graphStatus GRAPH_PARAM_INVALID = 50331649;

}

#endif //GE_ERROR_CODES_H
