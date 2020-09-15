/*
 * Copyright 1993-2019 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#ifndef NV_INFER_PLUGIN_H
#define NV_INFER_PLUGIN_H

#include "NvInfer.h"
#include "NvInferPluginUtils.h"
//!
//! \file NvInferPlugin.h
//!
//! This is the API for the Nvidia provided TensorRT plugins.
//!

namespace nvinfer1
{

namespace plugin
{
//!
//! \class INvPlugin
//!
//! \brief Common interface for the Nvidia created plugins.
//!
//! This class provides a common subset of functionality that is used
//! to provide distinguish the Nvidia created plugins. Each plugin provides a
//! function to validate the parameter options and create the plugin
//! object.
//!
class INvPlugin : public IPlugin
{
public:
    //!
    //! \brief Get the parameter plugin ID.
    //!
    //! \return The ID of the plugin.
    //!
    virtual PluginType getPluginType() const TRTNOEXCEPT = 0;

    //!
    //! \brief Get the name of the plugin from the ID
    //!
    //! \return The name of the plugin specified by \p id. Return nullptr if invalid ID is specified.
    //!
    //! The valid \p id values are ranged [0, numPlugins()).
    //!
    virtual const char* getName() const TRTNOEXCEPT = 0;

    //!
    //! \brief Destroy the plugin.
    //!
    //! The valid \p id values are ranged [0, numPlugins()).
    //!
    virtual void destroy() TRTNOEXCEPT = 0;

protected:
    ~INvPlugin() TRTNOEXCEPT {}
}; // INvPlugin

//!
//! \param featureStride Feature stride.
//! \param preNmsTop Number of proposals to keep before applying NMS.
//! \param nmsMaxOut Number of remaining proposals after applying NMS.
//! \param iouThreshold IoU threshold.
//! \param minBoxSize Minimum allowed bounding box size before scaling.
//! \param spatialScale Spatial scale between the input image and the last feature map.
//! \param pooling Spatial dimensions of pooled ROIs.
//! \param anchorRatios Aspect ratios for generating anchor windows.
//! \param anchorScales Scales for generating anchor windows.
//! \brief Create a plugin layer that fuses the RPN and ROI pooling using user-defined parameters.
//!
//! \return Returns a FasterRCNN fused RPN+ROI pooling plugin. Returns nullptr on invalid inputs.
//!
//! \see INvPlugin
//! \deprecated. This plugin is superseded by createRPNROIPlugin()
//!
TRT_DEPRECATED_API INvPlugin* createFasterRCNNPlugin(int featureStride, int preNmsTop,
                                              int nmsMaxOut, float iouThreshold, float minBoxSize,
                                              float spatialScale, DimsHW pooling,
                                              Weights anchorRatios, Weights anchorScales);
TRT_DEPRECATED_API INvPlugin* createFasterRCNNPlugin(const void* data, size_t length);

//!
//! \brief The Normalize plugin layer normalizes the input to have L2 norm of 1 with scale learnable.
//! \param scales Scale weights that are applied to the output tensor.
//! \param acrossSpatial Whether to compute the norm over adjacent channels (acrossSpatial is true) or nearby spatial locations (within channel in which case acrossSpatial is false).
//! \param channelShared Whether the scale weight(s) is shared across channels.
//! \param eps Epsilon for not diviiding by zero.
//! \deprecated. This plugin is superseded by createNormalizePlugin()
//!
TRT_DEPRECATED_API INvPlugin* createSSDNormalizePlugin(const Weights* scales, bool acrossSpatial, bool channelShared, float eps);
TRT_DEPRECATED_API INvPlugin* createSSDNormalizePlugin(const void* data, size_t length);


//!
//! \param permuteOrder The new orders that are used to permute the data.
//! \deprecated. Please use the TensorRT Shuffle layer for Permute operation
//!
TRT_DEPRECATED_API INvPlugin* createSSDPermutePlugin(Quadruple permuteOrder);
TRT_DEPRECATED_API INvPlugin* createSSDPermutePlugin(const void* data, size_t length);


//!
//! \param param Set of parameters for creating the PriorBox plugin layer.
//! \deprecated. This plugin is superseded by createPriorBoxPlugin()
//!
TRT_DEPRECATED_API INvPlugin* createSSDPriorBoxPlugin(PriorBoxParameters param);
TRT_DEPRECATED_API INvPlugin* createSSDPriorBoxPlugin(const void* data, size_t length);

//!
//! \brief The Grid Anchor Generator plugin layer generates the prior boxes of
//! designated sizes and aspect ratios across all dimensions (H x W) for all feature maps.
//! GridAnchorParameters defines a set of parameters for creating the GridAnchorGenerator plugin layer.
//! \deprecated. This plugin is superseded by createAnchorGeneratorPlugin()
//!
TRT_DEPRECATED_API INvPlugin* createSSDAnchorGeneratorPlugin(GridAnchorParameters* param, int numLayers);
TRT_DEPRECATED_API INvPlugin* createSSDAnchorGeneratorPlugin(const void* data, size_t length);


//!
//! \param param Set of parameters for creating the DetectionOutput plugin layer.
//! \deprecated. This plugin is superseded by createNMSPlugin()
//!
TRT_DEPRECATED_API INvPlugin* createSSDDetectionOutputPlugin(DetectionOutputParameters param);
TRT_DEPRECATED_API INvPlugin* createSSDDetectionOutputPlugin(const void* data, size_t length);

//!
//! \brief The Concat plugin layer basically performs the concatention for 4D tensors. Unlike the Concatenation layer in early version of TensorRT,
//! it allows the user to specify the axis along which to concatenate. The axis can be 1 (across channel), 2 (across H), or 3 (across W).
//! More particularly, this Concat plugin layer also implements the "ignoring the batch dimension" switch. If turned on, all the input tensors will be treated as if their batch sizes were 1.
//! \param concatAxis Axis along which to concatenate. Can't be the "N" dimension.
//! \param ignoreBatch If true, all the input tensors will be treated as if their batch sizes were 1.
//! \deprecated. This plugin is superseded by native TensorRT concatenation layer
//!
TRT_DEPRECATED_API INvPlugin* createConcatPlugin(int concatAxis, bool ignoreBatch);
TRT_DEPRECATED_API INvPlugin* createConcatPlugin(const void* data, size_t length);

//!
//! \brief The PReLu plugin layer performs leaky ReLU for 4D tensors. Give an input value x, the PReLU layer computes the output as x if x > 0
//!  and negative_slope //! x if x <= 0.
//! \param negSlope Negative_slope value.
//! \deprecated. This plugin is superseded by createLReLUPlugin()
//!
TRT_DEPRECATED_API INvPlugin* createPReLUPlugin(float negSlope);
TRT_DEPRECATED_API INvPlugin* createPReLUPlugin(const void* data, size_t length);

//!
//! \brief The Reorg plugin layer maps the 512x26x26 feature map onto a 2048x13x13 feature map, so that it can be concatenated with the feature maps at 13x13 resolution.
//! \param stride Strides in H and W.
//! \deprecated. This plugin is superseded by createReorgPlugin()
//!
TRT_DEPRECATED_API INvPlugin* createYOLOReorgPlugin(int stride);
TRT_DEPRECATED_API INvPlugin* createYOLOReorgPlugin(const void* data, size_t length);

TRT_DEPRECATED_API INvPlugin* createYOLORegionPlugin(RegionParameters params);
TRT_DEPRECATED_API INvPlugin* createYOLORegionPlugin(const void* data, size_t length);


} // end plugin namespace
} // end nvinfer1 namespace

extern "C"
{
//!
//! \brief Create a plugin layer that fuses the RPN and ROI pooling using user-defined parameters.
//! Registered plugin type "RPROI_TRT". Registered plugin version "1".
//! \param featureStride Feature stride.
//! \param preNmsTop Number of proposals to keep before applying NMS.
//! \param nmsMaxOut Number of remaining proposals after applying NMS.
//! \param iouThreshold IoU threshold.
//! \param minBoxSize Minimum allowed bounding box size before scaling.
//! \param spatialScale Spatial scale between the input image and the last feature map.
//! \param pooling Spatial dimensions of pooled ROIs.
//! \param anchorRatios Aspect ratios for generating anchor windows.
//! \param anchorScales Scales for generating anchor windows.
//!
//! \return Returns a FasterRCNN fused RPN+ROI pooling plugin. Returns nullptr on invalid inputs.
//!
TENSORRTAPI nvinfer1::IPluginV2* createRPNROIPlugin(int featureStride, int preNmsTop,
                                                                int nmsMaxOut, float iouThreshold, float minBoxSize,
                                                                float spatialScale, nvinfer1::DimsHW pooling,
                                                                nvinfer1::Weights anchorRatios, nvinfer1::Weights anchorScales);

//!
//! \brief The Normalize plugin layer normalizes the input to have L2 norm of 1 with scale learnable.
//! Registered plugin type "Normalize_TRT". Registered plugin version "1".
//! \param scales Scale weights that are applied to the output tensor.
//! \param acrossSpatial Whether to compute the norm over adjacent channels (acrossSpatial is true) or nearby spatial locations (within channel in which case acrossSpatial is false).
//! \param channelShared Whether the scale weight(s) is shared across channels.
//! \param eps Epsilon for not diviiding by zero.
//!
TENSORRTAPI nvinfer1::IPluginV2* createNormalizePlugin(const nvinfer1::Weights* scales, bool acrossSpatial, bool channelShared, float eps);

//!
//! \brief The PriorBox plugin layer generates the prior boxes of designated sizes and aspect ratios across all dimensions (H x W).
//! PriorBoxParameters defines a set of parameters for creating the PriorBox plugin layer.
//! Registered plugin type "PriorBox_TRT". Registered plugin version "1".
//!
TENSORRTAPI nvinfer1::IPluginV2* createPriorBoxPlugin(nvinfer1::plugin::PriorBoxParameters param);

//!
//! \brief The Grid Anchor Generator plugin layer generates the prior boxes of
//! designated sizes and aspect ratios across all dimensions (H x W) for all feature maps.
//! GridAnchorParameters defines a set of parameters for creating the GridAnchorGenerator plugin layer.
//! Registered plugin type "GridAnchor_TRT". Registered plugin version "1".
//!
TENSORRTAPI nvinfer1::IPluginV2* createAnchorGeneratorPlugin(nvinfer1::plugin::GridAnchorParameters* param, int numLayers);

//!
//! \brief The DetectionOutput plugin layer generates the detection output based on location and confidence predictions by doing non maximum suppression.
//! DetectionOutputParameters defines a set of parameters for creating the DetectionOutput plugin layer.
//! Registered plugin type "NMS_TRT". Registered plugin version "1".
//!
TENSORRTAPI nvinfer1::IPluginV2* createNMSPlugin(nvinfer1::plugin::DetectionOutputParameters param);

//!
//! \brief The LReLu plugin layer performs leaky ReLU for 4D tensors. Give an input value x, the PReLU layer computes the output as x if x > 0 and negative_slope //! x if x <= 0.
//! Registered plugin type "LReLU_TRT". Registered plugin version "1".
//! \param negSlope Negative_slope value.
//!
TRT_DEPRECATED_API nvinfer1::IPluginV2* createLReLUPlugin(float negSlope);

//!
//! \brief The Reorg plugin reshapes input of shape CxHxW into a (C*stride*stride)x(H/stride)x(W/stride) shape, used in YOLOv2.
//! It does that by taking 1 x stride x stride slices from tensor and flattening them into (stridexstride) x 1 x 1 shape.
//! Registered plugin type "Reorg_TRT". Registered plugin version "1".
//! \param stride Strides in H and W, it should divide both H and W. Also stride * stride should be less than or equal to C.
//!
TENSORRTAPI nvinfer1::IPluginV2* createReorgPlugin(int stride);

//!
//! \brief The Region plugin layer performs region proposal calculation: generate 5 bounding boxes per cell (for yolo9000, generate 3 bounding boxes per cell).
//! For each box, calculating its probablities of objects detections from 80 pre-defined classifications (yolo9000 has 9416 pre-defined classifications,
//! and these 9416 items are organized as work-tree structure).
//! RegionParameters defines a set of parameters for creating the Region plugin layer.
//! Registered plugin type "Region_TRT". Registered plugin version "1".
//!
TENSORRTAPI nvinfer1::IPluginV2* createRegionPlugin(nvinfer1::plugin::RegionParameters params);

//!
//! \brief The Clip Plugin performs a clip operation on the input tensor. It
//! clips the tensor values to a specified min and max. Any value less than clipMin are set to clipMin.
//! Any values greater than clipMax are set to clipMax. For example, this plugin can be used
//! to perform a Relu6 operation by specifying clipMin=0.0 and clipMax=6.0
//! Registered plugin type "Clip_TRT". Registered plugin version "1".
//! \param layerName The name of the TensorRT layer.
//! \param clipMin The minimum value to clip to.
//! \param clipMax The maximum value to clip to.
//!
TRT_DEPRECATED_API nvinfer1::IPluginV2* createClipPlugin(const char* layerName, float clipMin, float clipMax);

//!
//! \brief The BatchedNMS Plugin performs non_max_suppression on the input boxes, per batch, across all classes.
//! It greedily selects a subset of bounding boxes in descending order of
//! score. Prunes away boxes that have a high intersection-over-union (IOU)
//! overlap with previously selected boxes. Bounding boxes are supplied as [y1, x1, y2, x2],
//! where (y1, x1) and (y2, x2) are the coordinates of any
//! diagonal pair of box corners and the coordinates can be provided as normalized
//! (i.e., lying in the interval [0, 1]) or absolute.
//! The plugin expects two inputs.
//! Input0 is expected to be 4-D float boxes tensor of shape [batch_size, num_boxes,
//! q, 4], where q can be either 1 (if shareLocation is true) or num_classes.
//! Input1 is expected to be a 3-D float scores tensor of shape [batch_size, num_boxes, num_classes]
//! representing a single score corresponding to each box.
//! The plugin returns four outputs.
//! num_detections : A [batch_size] int32 tensor indicating the number of valid
//! detections per batch item. Can be less than keepTopK. Only the top num_detections[i] entries in
//! nmsed_boxes[i], nmsed_scores[i] and nmsed_classes[i] are valid.
//! nmsed_boxes : A [batch_size, max_detections, 4] float32 tensor containing
//! the co-ordinates of non-max suppressed boxes.
//! nmsed_scores : A [batch_size, max_detections] float32 tensor containing the
//! scores for the boxes.
//! nmsed_classes :  A [batch_size, max_detections] float32 tensor containing the
//! classes for the boxes.
//!
//! Registered plugin type "BatchedNMS_TRT". Registered plugin version "1".
//!
//! The batched NMS plugin can require a lot of workspace due to intermediate buffer usage. To get the
//! estimated workspace size for the plugin for a batch size, use the API `plugin->getWorkspaceSize(batchSize)`.
//!
TENSORRTAPI nvinfer1::IPluginV2* createBatchedNMSPlugin(nvinfer1::plugin::NMSParameters param);

//!
//! \brief The Split Plugin performs a split operation on the input tensor. It
//! splits the input tensor into several output tensors, each of a length corresponding to output_lengths.
//! The split occurs along the axis specified by axis.
//! \param axis The axis to split on.
//! \param output_lengths The lengths of the output tensors.
//! \param noutput The number of output tensors.
//!
TENSORRTAPI nvinfer1::IPluginV2* createSplitPlugin(int axis, int* output_lengths, int noutput);

//!
//! \brief The Instance Normalization Plugin computes the instance normalization of an input tensor.
//! The instance normalization is calculated as found in the paper https://arxiv.org/abs/1607.08022.
//! The calculation is y = scale * (x - mean) / sqrt(variance + epsilon) + bias where mean and variance
//! are computed per instance per channel.
//! \param epsilon The epsilon value to use to avoid division by zero.
//! \param scale_weights The input 1-dimensional scale weights of size C to scale.
//! \param bias_weights The input 1-dimensional bias weights of size C to offset.
//!
TENSORRTAPI nvinfer1::IPluginV2* createInstanceNormalizationPlugin(float epsilon, nvinfer1::Weights scale_weights, nvinfer1::Weights bias_weights);

//!
//! \brief Initialize and register all the existing TensorRT plugins to the Plugin Registry with an optional namespace.
//! The plugin library author should ensure that this function name is unique to the library.
//! This function should be called once before accessing the Plugin Registry.
//! \param logger Logger object to print plugin registration information
//! \param libNamespace Namespace used to register all the plugins in this library
//!
TENSORRTAPI bool initLibNvInferPlugins(void* logger, const char* libNamespace);

} // extern "C"

#endif // NV_INFER_PLUGIN_H
