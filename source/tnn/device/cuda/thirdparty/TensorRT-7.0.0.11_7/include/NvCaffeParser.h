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

#ifndef NV_CAFFE_PARSER_H
#define NV_CAFFE_PARSER_H

#include "NvInfer.h"

//!
//! \namespace nvcaffeparser1
//!
//! \brief The TensorRT Caffe parser API namespace.
//!
namespace nvcaffeparser1
{

//!
//! \class IBlobNameToTensor
//!
//! \brief Object used to store and query Tensors after they have been extracted from a Caffe model using the ICaffeParser.
//!
//! \note The lifetime of IBlobNameToTensor is the same as the lifetime of its parent ICaffeParser.
//!
//! \see nvcaffeparser1::ICaffeParser
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IBlobNameToTensor
{
public:
    //! \brief Given a blob name, returns a pointer to a ITensor object.
    //!
    //! \param name Caffe blob name for which the user wants the corresponding ITensor.
    //!
    //! \return ITensor* corresponding to the queried name. If no such ITensor exists, then nullptr is returned.
    //!
    virtual nvinfer1::ITensor* find(const char* name) const TRTNOEXCEPT = 0;

protected:
    virtual ~IBlobNameToTensor() {}
};

//!
//! \class IBinaryProtoBlob
//!
//! \brief Object used to store and query data extracted from a binaryproto file using the ICaffeParser.
//!
//! \see nvcaffeparser1::ICaffeParser
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IBinaryProtoBlob
{
public:
    virtual const void* getData() TRTNOEXCEPT = 0;
    virtual nvinfer1::DimsNCHW getDimensions() TRTNOEXCEPT = 0;
    virtual nvinfer1::DataType getDataType() TRTNOEXCEPT = 0;
    virtual void destroy() TRTNOEXCEPT = 0;

protected:
    virtual ~IBinaryProtoBlob() {}
};

//!
//! \class IPluginFactory
//!
//! \brief Plugin factory used to configure plugins.
//!
class IPluginFactory
{
public:
    //!
    //! \brief A user implemented function that determines if a layer configuration is provided by an IPlugin.
    //!
    //! \param layerName Name of the layer which the user wishes to validate.
    //!
    virtual bool isPlugin(const char* layerName) TRTNOEXCEPT = 0;

    //!
    //! \brief Creates a plugin.
    //!
    //! \param layerName Name of layer associated with the plugin.
    //! \param weights Weights used for the layer.
    //! \param nbWeights Number of weights.
    //!
    virtual nvinfer1::IPlugin* createPlugin(const char* layerName, const nvinfer1::Weights* weights, int nbWeights) TRTNOEXCEPT = 0;

    virtual ~IPluginFactory() {}
};

//!
//! \class IPluginFactoryExt
//!
//! \brief Plugin factory used to configure plugins with added support for TRT versioning.
//!
class IPluginFactoryExt : public IPluginFactory
{
public:
    virtual int getVersion() const TRTNOEXCEPT
    {
        return NV_TENSORRT_VERSION;
    }

    //!
    //! \brief A user implemented function that determines if a layer configuration is provided by an IPluginExt.
    //!
    //! \param layerName Name of the layer which the user wishes to validate.
    //!
    virtual bool isPluginExt(const char* layerName) TRTNOEXCEPT = 0;
};

//!
//! \class IPluginFactoryV2
//!
//! \brief Plugin factory used to configure plugins.
//!
class IPluginFactoryV2
{
public:
    //!
    //! \brief A user implemented function that determines if a layer configuration is provided by an IPluginV2.
    //!
    //! \param layerName Name of the layer which the user wishes to validate.
    //!
    virtual bool isPluginV2(const char* layerName) TRTNOEXCEPT = 0;

    //!
    //! \brief Creates a plugin.
    //!
    //! \param layerName Name of layer associated with the plugin.
    //! \param weights Weights used for the layer.
    //! \param nbWeights Number of weights.
    //! \param libNamespace Library Namespace associated with the plugin object
    //!
    virtual nvinfer1::IPluginV2* createPlugin(const char* layerName, const nvinfer1::Weights* weights, int nbWeights, const char* libNamespace = "") TRTNOEXCEPT = 0;

    virtual ~IPluginFactoryV2() {}
};
//!
//! \class ICaffeParser
//!
//! \brief Class used for parsing Caffe models.
//!
//! Allows users to export models trained using Caffe to TRT.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class ICaffeParser
{
public:
    //!
    //! \brief Parse a prototxt file and a binaryproto Caffe model to extract
    //!   network definition and weights associated with the network, respectively.
    //!
    //! \param deploy The plain text, prototxt file used to define the network definition.
    //! \param model The binaryproto Caffe model that contains the weights associated with the network.
    //! \param network Network in which the CaffeParser will fill the layers.
    //! \param weightType The type to which the weights will transformed.
    //!
    //! \return A pointer to an IBlobNameToTensor object that contains the extracted data.
    //!
    //! \see nvcaffeparser1::IBlobNameToTensor
    //!
    virtual const IBlobNameToTensor* parse(const char* deploy,
                                           const char* model,
                                           nvinfer1::INetworkDefinition& network,
                                           nvinfer1::DataType weightType) TRTNOEXCEPT = 0;

    //!
    //! \brief Parse a deploy prototxt a binaryproto Caffe model from memory buffers to extract
    //!   network definition and weights associated with the network, respectively.
    //!
    //! \param deployBuffer The plain text deploy prototxt used to define the network definition.
    //! \param deployLength The length of the deploy buffer.
    //! \param modelBuffer The binaryproto Caffe memory buffer that contains the weights associated with the network.
    //! \param modelLength The length of the model buffer.
    //! \param network Network in which the CaffeParser will fill the layers.
    //! \param weightType The type to which the weights will transformed.
    //!
    //! \return A pointer to an IBlobNameToTensor object that contains the extracted data.
    //!
    //! \see nvcaffeparser1::IBlobNameToTensor
    //!
    virtual const IBlobNameToTensor* parseBuffers(const char* deployBuffer,
                                                  std::size_t deployLength,
                                                  const char* modelBuffer,
                                                  std::size_t modelLength,
                                                  nvinfer1::INetworkDefinition& network,
                                                  nvinfer1::DataType weightType) TRTNOEXCEPT = 0;

    //!
    //! \brief Parse and extract data stored in binaryproto file.
    //!
    //! The binaryproto file contains data stored in a binary blob. parseBinaryProto() converts it
    //! to an IBinaryProtoBlob object which gives the user access to the data and meta-data about data.
    //!
    //! \param fileName Path to file containing binary proto.
    //!
    //! \return A pointer to an IBinaryProtoBlob object that contains the extracted data.
    //!
    //! \see nvcaffeparser1::IBinaryProtoBlob
    //!
    virtual IBinaryProtoBlob* parseBinaryProto(const char* fileName) TRTNOEXCEPT = 0;

    //!
    //! \brief Set buffer size for the parsing and storage of the learned model.
    //!
    //! \param size The size of the buffer specified as the number of bytes.
    //!
    //! \note  Default size is 2^30 bytes.
    //!
    virtual void setProtobufBufferSize(size_t size) TRTNOEXCEPT = 0;

    //!
    //! \brief Set the IPluginFactory used to create the user defined plugins.
    //!
    //! \param factory Pointer to an instance of the user implmentation of IPluginFactory.
    //!
    virtual void setPluginFactory(IPluginFactory* factory) TRTNOEXCEPT = 0;

    //!
    //! \brief Set the IPluginFactoryExt used to create the user defined pluginExts.
    //!
    //! \param factory Pointer to an instance of the user implmentation of IPluginFactoryExt.
    //!
    virtual void setPluginFactoryExt(IPluginFactoryExt* factory) TRTNOEXCEPT = 0;

    //!
    //! \brief Destroy this ICaffeParser object.
    //!
    virtual void destroy() TRTNOEXCEPT = 0;

    //!
    //! \brief Set the IPluginFactoryV2 used to create the user defined pluginV2 objects.
    //!
    //! \param factory Pointer to an instance of the user implmentation of IPluginFactoryV2.
    //!
    virtual void setPluginFactoryV2(IPluginFactoryV2* factory) TRTNOEXCEPT = 0;

    //!
    //! \brief Set the namespace used to lookup and create plugins in the network.
    //!
    virtual void setPluginNamespace(const char* libNamespace) TRTNOEXCEPT = 0;

protected:
    virtual ~ICaffeParser() {}

public:
    //!
    //! \brief Set the ErrorRecorder for this interface
    //!
    //! Assigns the ErrorRecorder to this interface. The ErrorRecorder will track all errors during execution.
    //! This function will call incRefCount of the registered ErrorRecorder at least once. Setting
    //! recorder to nullptr unregisters the recorder with the interface, resulting in a call to decRefCount if
    //! a recorder has been registered.
    //!
    //! \param recorder The error recorder to register with this interface.
    //!
    //! \see getErrorRecorder
    //!
    virtual void setErrorRecorder(nvinfer1::IErrorRecorder* recorder) TRTNOEXCEPT = 0;

    //!
    //! \brief get the ErrorRecorder assigned to this interface.
    //!
    //! Retrieves the assigned error recorder object for the given class. A default error recorder does not exist,
    //! so a nullptr will be returned if setErrorRecorder has not been called.
    //!
    //! \return A pointer to the IErrorRecorder object that has been registered.
    //!
    //! \see setErrorRecorder
    //!
    virtual nvinfer1::IErrorRecorder* getErrorRecorder() const TRTNOEXCEPT = 0;
};

//!
//! \brief Creates a ICaffeParser object.
//!
//! \return A pointer to the ICaffeParser object is returned.
//!
//! \see nvcaffeparser1::ICaffeParser
//!
TENSORRTAPI ICaffeParser* createCaffeParser() TRTNOEXCEPT;

//!
//! \brief Shuts down protocol buffers library.
//!
//! \note No part of the protocol buffers library can be used after this function is called.
//!
TENSORRTAPI void shutdownProtobufLibrary() TRTNOEXCEPT;
} // namespace nvcaffeparser1

//!
//! Internal C entry point for creating ICaffeParser.
//! @private
//!
extern "C" TENSORRTAPI void* createNvCaffeParser_INTERNAL();
#endif
