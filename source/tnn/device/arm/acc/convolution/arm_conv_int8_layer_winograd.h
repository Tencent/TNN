//  Copyright © 2019 tencent. All rights reserved.

#ifndef TNN_SOURCE_DEVICE_ARM_ARM_CONV_INT8_LAYER_ACC_WINOGRAD_H_
#define TNN_SOURCE_DEVICE_ARM_ARM_CONV_INT8_LAYER_ACC_WINOGRAD_H_

#include "tnn/device/arm/acc/convolution/arm_conv_int8_layer_common.h"

namespace TNN_NS {
// @brief conv layer cpu acc
class ArmConvInt8LayerWinograd : public ArmConvInt8LayerCommon {
public:
    /**
     * @brief init layer with param, resouce, intput blobs and output blobs.
     * @param context cpu context
     * @param param    layer param
     * @param resource  layer resouce
     * @param inputs    input blobs
     * @param outputs   output blobs
     */
    Status Init(Context *context, LayerParam *param, LayerResource *resource,
                const std::vector<Blob *> &input,
                const std::vector<Blob *> &outputs);

    // @brief virtual destrcutor
    virtual ~ArmConvInt8LayerWinograd();

    /**
     * @brief input or output blobs reshape.
     * @param inputs    input blobs
     * @param outputs   output blobs
     * @return reshape result
     */
    virtual Status Reshape(const std::vector<Blob *> &inputs,
                           const std::vector<Blob *> &outputs);

    /**
     * @brief layer forward
     * @param inputs    input blobs
     * @param outputs   output blobs
     * @return execution result
     */
    virtual Status DoForward(const std::vector<Blob *> &inputs,
                           const std::vector<Blob *> &outputs);

    /**
     * @brief allocate MTLBuffer for weights
     * @param inputs    input blobs
     * @param outputs   output blobs
     */
    virtual Status allocateBufferWeight(const std::vector<Blob *> &inputs,
                                        const std::vector<Blob *> &outputs);

    /**
     * @brief allocate MTLBuffer for bias
     * @param inputs    input blobs
     * @param outputs   output blobs
     */
    virtual Status allocateBufferBias(const std::vector<Blob *> &inputs,
                                      const std::vector<Blob *> &outputs);
    /**
     * @brief allocate MTLBuffer for scale
     * @param inputs    input blobs
     * @param outputs   output blobs
     */
    virtual Status allocateBufferScale(const std::vector<Blob *> &inputs,
                                       const std::vector<Blob *> &outputs);

    /**
     * @brief allocate MTLBuffer for winograd temp io
     * @param inputs    input blobs
     * @param outputs   output blobs
     */
    virtual Status allocateBufferWinoBuf(const std::vector<Blob *> &inputs,
                                         const std::vector<Blob *> &outputs);
    /**
     * @brief layer forward
     * @param param    convolution para
     * @param inputs    input blobs
     * @param outputs   output blobs
     * @return implement is prefered
     */
    static bool isPrefered(ConvLayerParam *param,
                           const std::vector<Blob *> &inputs,
                           const std::vector<Blob *> &outputs);

protected:
    RawBuffer buffer_weight_;
    RawBuffer buffer_srcwino_;
    RawBuffer buffer_srcpad_;
    RawBuffer buffer_dstwino_;
    RawBuffer buffer_bias_;
    RawBuffer buffer_scale_;
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_DEVICE_ARM_ARM_CONV_INT8_LAYER_ACC_WINOGRAD_H_
