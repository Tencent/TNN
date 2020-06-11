/**
 * Copyright (C)  2019. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file types.h
 *
 * @brief
 *
 * @version 1.0
 *
 */

#ifndef GE_PUBLIC_TYPES_H
#define GE_PUBLIC_TYPES_H

#include <memory>
#include <vector>
#include <atomic>


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

    enum DataType {
        DT_UNDEFINED = 16,   // Used to indicate a DataType field has not been set.
        DT_FLOAT = 0,           // float type
        DT_FLOAT16 = 1,         // fp16 type
        DT_INT8 = 2,            // int8 type
        DT_INT16 = 6,     // int16 type
        DT_UINT16 = 7,      // uint16 type
        DT_UINT8 = 4,           // uint8 type
        DT_INT32 = 3,           //
        DT_INT64 = 9,           // int64 type
        DT_UINT32 = 8,          // unsigned int32
        DT_UINT64 = 10,          // unsigned int64
        DT_BOOL = 12,            // bool type
        DT_DOUBLE = 11,          // double type
        DT_DUAL = 13,              /**< dual output type */
        DT_DUAL_SUB_INT8 = 14,    /**< dual output int8 type */
        DT_DUAL_SUB_UINT8 = 15,    /**< dual output uint8 type */
    };

    enum Format {
        FORMAT_NCHW = 0,         /**< NCHW */
        FORMAT_NHWC,             /**< NHWC */
        FORMAT_ND,               /**< Nd Tensor */
        FORMAT_NC1HWC0,          /**< NC1HWC0 */
        FORMAT_FRACTAL_Z,        /**< FRACTAL_Z */
        FORMAT_NC1C0HWPAD,
        FORMAT_NHWC1C0,
        FORMAT_FSR_NCHW,
        FORMAT_FRACTAL_DECONV,
        FORMAT_C1HWNC0,
        FORMAT_FRACTAL_DECONV_TRANSPOSE,
        FORMAT_FRACTAL_DECONV_SP_STRIDE_TRANS,
        FORMAT_NC1HWC0_C04,  /**< NC1HWC0, C0 =4*/
        FORMAT_FRACTAL_Z_C04,/**< FRACZ格式，C0 =4 */
        FORMAT_CHWN,
        FORMAT_FRACTAL_DECONV_SP_STRIDE8_TRANS,
        FORMAT_HWCN,
        FORMAT_NC1KHKWHWC0, /** < KH,KW kernel h& kernel w maxpooling max output format*/
        FORMAT_BN_WEIGHT,
        FORMAT_FILTER_HWCK,     /* filter input tensor format */
        FORMAT_HASHTABLE_LOOKUP_LOOKUPS=20,
        FORMAT_HASHTABLE_LOOKUP_KEYS,
        FORMAT_HASHTABLE_LOOKUP_VALUE,
        FORMAT_HASHTABLE_LOOKUP_OUTPUT,
        FORMAT_HASHTABLE_LOOKUP_HITS=24,
        FORMAT_RESERVED

    };

    enum DeviceType {
        NPU = 0,
        CPU = 1,
//        DeviceType_INT_MIN_SENTINEL_DO_NOT_USE_ = ::google::protobuf::kint32min,
//        DeviceType_INT_MAX_SENTINEL_DO_NOT_USE_ = ::google::protobuf::kint32max
    };

    enum AnchorStatus {
        ANCHOR_SUSPEND = 0,         //dat null
        ANCHOR_CONST = 1,
        ANCHOR_DATA = 2,            //有效的
        ANCHOR_RESERVED = 3
    };

    struct TensorType {
        explicit TensorType(DataType dt)
        {
            dt_vec_.push_back(dt);
        };

        TensorType(std::initializer_list<DataType> types)
        {
            dt_vec_ = types;
        };

        static TensorType ALL()
        {
            return TensorType{DT_FLOAT,
                              DT_FLOAT16,
                              DT_INT8,
                              DT_INT16,
                              DT_UINT16,
                              DT_UINT8,
                              DT_INT32,
                              DT_INT64,
                              DT_UINT32,
                              DT_UINT64,
                              DT_BOOL,
                              DT_DOUBLE,
                              DT_DUAL,
                              DT_DUAL_SUB_INT8,
                              DT_DUAL_SUB_UINT8};
        }

        static TensorType FLOAT()
        {
            return TensorType{DT_FLOAT, DT_FLOAT16};
        }

        std::vector<DataType> dt_vec_;
    };
}


#endif //GE_PUBLIC_TYPES_H
