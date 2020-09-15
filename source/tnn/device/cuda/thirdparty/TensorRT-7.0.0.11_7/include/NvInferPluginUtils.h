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

#ifndef NV_INFER_PLUGIN_UTILS_H
#define NV_INFER_PLUGIN_UTILS_H
#include "NvInferRuntimeCommon.h"
//!
//! \file NvPluginUtils.h
//!
//! This is the API for the Nvidia provided TensorRT plugin utilities.
//! It lists all the parameters utilized by the TensorRT plugins.
//!

namespace nvinfer1
{
//!
//! \enum PluginType
//!
//! \brief The type values for the various plugins.
//!
//! \see INvPlugin::getPluginType()
//!
enum class PluginType : int
{
    kFASTERRCNN = 0,         //!< FasterRCNN fused plugin (RPN + ROI pooling).
    kNORMALIZE = 1,          //!< Normalize plugin.
    kPERMUTE = 2,            //!< Permute plugin.
    kPRIORBOX = 3,           //!< PriorBox plugin.
    kSSDDETECTIONOUTPUT = 4, //!< SSD DetectionOutput plugin.
    kCONCAT = 5,             //!< Concat plugin.
    kPRELU = 6,              //!< YOLO PReLU Plugin.
    kYOLOREORG = 7,          //!< YOLO Reorg Plugin.
    kYOLOREGION = 8,         //!< YOLO Region Plugin.
    kANCHORGENERATOR = 9,    //!< SSD Grid Anchor Generator.
};

//!< Maximum number of elements in PluginType enum. \see PluginType
template <>
constexpr inline int EnumMax<PluginType>()
{
    return 10;
}

namespace plugin
{

//!
//! \brief The Permute plugin layer permutes the input tensor by changing the memory order of the data.
//! Quadruple defines a structure that contains an array of 4 integers. They can represent the permute orders or the strides in each dimension.
//!
typedef struct
{
    int data[4];
} Quadruple;


//!
//! \brief The PriorBox plugin layer generates the prior boxes of designated sizes and aspect ratios across all dimensions (H x W).
//! PriorBoxParameters defines a set of parameters for creating the PriorBox plugin layer.
//! It contains:
//! \param minSize Minimum box size in pixels. Can not be nullptr.
//! \param maxSize Maximum box size in pixels. Can be nullptr.
//! \param aspectRatios Aspect ratios of the boxes. Can be nullptr.
//! \param numMinSize Number of elements in minSize. Must be larger than 0.
//! \param numMaxSize Number of elements in maxSize. Can be 0 or same as numMinSize.
//! \param numAspectRatios Number of elements in aspectRatios. Can be 0.
//! \param flip If true, will flip each aspect ratio. For example, if there is aspect ratio "r", the aspect ratio "1.0/r" will be generated as well.
//! \param clip If true, will clip the prior so that it is within [0,1].
//! \param variance Variance for adjusting the prior boxes.
//! \param imgH Image height. If 0, then the H dimension of the data tensor will be used.
//! \param imgW Image width. If 0, then the W dimension of the data tensor will be used.
//! \param stepH Step in H. If 0, then (float)imgH/h will be used where h is the H dimension of the 1st input tensor.
//! \param stepW Step in W. If 0, then (float)imgW/w will be used where w is the W dimension of the 1st input tensor.
//! \param offset Offset to the top left corner of each cell.
//!
struct PriorBoxParameters
{
    float *minSize, *maxSize, *aspectRatios;
    int numMinSize, numMaxSize, numAspectRatios;
    bool flip;
    bool clip;
    float variance[4];
    int imgH, imgW;
    float stepH, stepW;
    float offset;
};

//!
//! \brief RPROIParams is used to create the RPROIPlugin instance.
//! It contains:
//! \param poolingH Height of the output in pixels after ROI pooling on feature map.
//! \param poolingW Width of the output in pixels after ROI pooling on feature map.
//! \param featureStride Feature stride; ratio of input image size to feature map size. Assuming that max pooling layers in neural network use square filters.
//! \param preNmsTop Number of proposals to keep before applying NMS.
//! \param nmsMaxOut Number of remaining proposals after applying NMS.
//! \param anchorsRatioCount Number of anchor box ratios.
//! \param anchorsScaleCount Number of anchor box scales.
//! \param iouThreshold IoU (Intersection over Union) threshold used for the NMS step.
//! \param minBoxSize Minimum allowed bounding box size before scaling, used for anchor box calculation.
//! \param spatialScale Spatial scale between the input image and the last feature map.
//!
struct RPROIParams
{
    int poolingH;
    int poolingW;
    int featureStride;
    int preNmsTop;
    int nmsMaxOut;
    int anchorsRatioCount;
    int anchorsScaleCount;
    float iouThreshold;
    float minBoxSize;
    float spatialScale;
};


//!
//! \brief The Anchor Generator plugin layer generates the prior boxes of designated sizes and aspect ratios across all dimensions (H x W).
//! GridAnchorParameters defines a set of parameters for creating the plugin layer for all feature maps.
//! It contains:
//! \param minScale Scale of anchors corresponding to finest resolution.
//! \param maxScale Scale of anchors corresponding to coarsest resolution.
//! \param aspectRatios List of aspect ratios to place on each grid point.
//! \param numAspectRatios Number of elements in aspectRatios.
//! \param H Height of feature map to generate anchors for.
//! \param W Width of feature map to generate anchors for.
//! \param variance Variance for adjusting the prior boxes.
//!
struct GridAnchorParameters
{
    float minSize, maxSize;
    float* aspectRatios;
    int numAspectRatios, H, W;
    float variance[4];
};

//!
//! \enum CodeTypeSSD
//! \brief The type of encoding used for decoding the bounding boxes and loc_data.
//!
enum class CodeTypeSSD : int
{
    CORNER = 0,      //!< Use box corners.
    CENTER_SIZE = 1, //!< Use box centers and size.
    CORNER_SIZE = 2, //!< Use box centers and size.
    TF_CENTER = 3    //!< Use box centers and size but flip x and y coordinates.
};

//!
//! \brief The DetectionOutput plugin layer generates the detection output based on location and confidence predictions by doing non maximum suppression.
//! This plugin first decodes the bounding boxes based on the anchors generated. It then performs non_max_suppression on the decoded bouding boxes.
//! DetectionOutputParameters defines a set of parameters for creating the DetectionOutput plugin layer.
//! It contains:
//! \param shareLocation If true, bounding box are shared among different classes.
//! \param varianceEncodedInTarget If true, variance is encoded in target. Otherwise we need to adjust the predicted offset accordingly.
//! \param backgroundLabelId Background label ID. If there is no background class, set it as -1.
//! \param numClasses Number of classes to be predicted.
//! \param topK Number of boxes per image with top confidence scores that are fed into the NMS algorithm.
//! \param keepTopK Number of total bounding boxes to be kept per image after NMS step.
//! \param confidenceThreshold Only consider detections whose confidences are larger than a threshold.
//! \param nmsThreshold Threshold to be used in NMS.
//! \param codeType Type of coding method for bbox.
//! \param inputOrder Specifies the order of inputs {loc_data, conf_data, priorbox_data}.
//! \param confSigmoid Set to true to calculate sigmoid of confidence scores.
//! \param isNormalized Set to true if bounding box data is normalized by the network.
//!
struct DetectionOutputParameters
{
    bool shareLocation, varianceEncodedInTarget;
    int backgroundLabelId, numClasses, topK, keepTopK;
    float confidenceThreshold, nmsThreshold;
    CodeTypeSSD codeType;
    int inputOrder[3];
    bool confSigmoid;
    bool isNormalized;
};


//!
//! \brief The Region plugin layer performs region proposal calculation: generate 5 bounding boxes per cell (for yolo9000, generate 3 bounding boxes per cell).
//! For each box, calculating its probablities of objects detections from 80 pre-defined classifications (yolo9000 has 9418 pre-defined classifications,
//! and these 9418 items are organized as work-tree structure).
//! RegionParameters defines a set of parameters for creating the Region plugin layer.
//! \param num Number of predicted bounding box for each grid cell.
//! \param coords Number of coordinates for a bounding box.
//! \param classes Number of classfications to be predicted.
//! \param softmaxTree When performing yolo9000, softmaxTree is helping to do softmax on confidence scores, for element to get the precise classfication through word-tree structured classfication definition.
//! \deprecated. This plugin is superseded by createRegionPlugin()
//!
TRT_DEPRECATED typedef struct
{
    int* leaf;
    int n;
    int* parent;
    int* child;
    int* group;
    char** name;

    int groups;
    int* groupSize;
    int* groupOffset;
} softmaxTree; // softmax tree

struct TRT_DEPRECATED RegionParameters
{
    int num;
    int coords;
    int classes;
    softmaxTree* smTree;
};

//!
//! \brief The NMSParameters are used by the BatchedNMSPlugin for performing
//! the non_max_suppression operation over boxes for object detection networks.
//! \param shareLocation If set to true, the boxes inputs are shared across all
//!        classes. If set to false, the boxes input should account for per class box data.
//! \param backgroundLabelId Label ID for the background class. If there is no background class, set it as -1
//! \param numClasses Number of classes in the network.
//! \param topK Number of bounding boxes to be fed into the NMS step.
//! \param keepTopK Number of total bounding boxes to be kept per image after NMS step.
//!        Should be less than or equal to the topK value.
//! \param scoreThreshold Scalar threshold for score (low scoring boxes are removed).
//! \param iouThreshold scalar threshold for IOU (new boxes that have high IOU overlap
//!        with previously selected boxes are removed).
//! \param isNormalized Set to false, if the box coordinates are not
//!        normalized, i.e. not in the range [0,1]. Defaults to false.
//!

struct NMSParameters
{
    bool shareLocation;
    int backgroundLabelId, numClasses, topK, keepTopK;
    float scoreThreshold, iouThreshold;
    bool isNormalized;
};

} // end plugin namespace
} // end nvinfer1 namespace
#endif
