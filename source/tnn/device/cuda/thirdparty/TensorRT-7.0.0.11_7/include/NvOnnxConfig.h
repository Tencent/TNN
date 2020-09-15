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

#ifndef NV_OnnxConfig_H
#define NV_OnnxConfig_H

#include "NvInfer.h"

namespace nvonnxparser
{

//!
//! \mainpage
//!
//! This is the API documentation for the Configuration Manager for Open Neural Network Exchange (ONNX) Parser for Nvidia TensorRT Inference Engine.
//! It provides information on individual functions, classes
//! and methods. Use the index on the left to navigate the documentation.
//!
//! Please see the accompanying user guide and samples for higher-level information and general advice on using ONNX Parser and TensorRT.
//!

//!
//! \file NvOnnxConfig.h
//!
//! This is the API file for the Configuration Manager for ONNX Parser for Nvidia TensorRT.
//!

//!
//! \class IOnnxConfig
//! \brief Configuration Manager Class.
//!
class IOnnxConfig
{
protected:
    virtual ~IOnnxConfig() {}

public:
    //!
    //! \typedef Verbosity
    //! \brief Defines Verbosity level.
    //!
    typedef int Verbosity;

    //!
    //! \brief Set the Model Data Type.
    //!
    //! Sets the Model DataType, one of the following: float -d 32 (default), half precision -d 16, and int8 -d 8 data types.
    //!
    //! \see getModelDtype()
    //!
    virtual void setModelDtype(const nvinfer1::DataType) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the Model Data Type.
    //!
    //! \return DataType nvinfer1::DataType
    //!
    //! \see setModelDtype() and #DataType
    //!
    virtual nvinfer1::DataType getModelDtype() const TRTNOEXCEPT = 0;

    //!
    //! \brief Get the Model FileName.
    //!
    //! \return Return the Model Filename, as a pointer to a NULL-terminated character sequence.
    //!
    //! \see setModelFileName()
    //!
    virtual const char* getModelFileName() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the Model File Name.
    //!
    //! The Model File name contains the Network Description in ONNX pb format.
    //!
    //! This method copies the name string.
    //!
    //! \param onnxFilename The name.
    //!
    //! \see getModelFileName()
    //!
    virtual void setModelFileName(const char* onnxFilename) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the Verbosity Level.
    //!
    //! \return The Verbosity Level.
    //!
    //! \see addVerbosity(), reduceVerbosity()
    //!
    virtual Verbosity getVerbosityLevel() const TRTNOEXCEPT = 0;

    //!
    //! \brief Increase the Verbosity Level.
    //!
    //! \return The Verbosity Level.
    //!
    //! \see addVerbosity(), reduceVerbosity(), setVerbosity(Verbosity)
    //!
    virtual void addVerbosity() TRTNOEXCEPT = 0;               //!< Increase verbosity Level.
    virtual void reduceVerbosity() TRTNOEXCEPT = 0;            //!< Decrease verbosity Level.
    virtual void setVerbosityLevel(Verbosity) TRTNOEXCEPT = 0; //!< Set to specific verbosity Level.

    //!
    //! \brief Returns the File Name of the Network Description as a Text File.
    //!
    //! \return Return the name of the file containing the network description converted to a plain text, used for debugging purposes.
    //!
    //! \see setTextFilename()
    //!
    virtual const char* getTextFileName() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the File Name of the Network Description as a Text File.
    //!
    //! This API allows setting a file name for the network description in plain text, equivalent of the ONNX protobuf.
    //!
    //! This method copies the name string.
    //!
    //! \param textFileName Name of the file.
    //!
    //! \see getTextFilename()
    //!
    virtual void setTextFileName(const char* textFileName) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the File Name of the Network Description as a Text File, including the weights.
    //!
    //! \return Return the name of the file containing the network description converted to a plain text, used for debugging purposes.
    //!
    //! \see setFullTextFilename()
    //!
    virtual const char* getFullTextFileName() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the File Name of the Network Description as a Text File, including the weights.
    //!
    //! This API allows setting a file name for the network description in plain text, equivalent of the ONNX protobuf.
    //!
    //! This method copies the name string.
    //!
    //! \param fullTextFileName Name of the file.
    //!
    //! \see getFullTextFilename()
    //!
    virtual void setFullTextFileName(const char* fullTextFileName) TRTNOEXCEPT = 0;

    //!
    //! \brief Get whether the layer information will be printed.
    //!
    //! \return Returns whether the layer information will be printed.
    //!
    //! \see setPrintLayerInfo()
    //!
    virtual bool getPrintLayerInfo() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set whether the layer information will be printed.
    //!
    //! \see getPrintLayerInfo()
    //!
    virtual void setPrintLayerInfo(bool) TRTNOEXCEPT = 0;

    //!
    //! \brief Destroy IOnnxConfig object.
    //!
    virtual void destroy() TRTNOEXCEPT = 0;

}; // class IOnnxConfig

TENSORRTAPI IOnnxConfig* createONNXConfig();

} // namespace nvonnxparser

#endif
