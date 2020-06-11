// =============================================================================
//
// Copyright (c) 2019 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
// =============================================================================

#ifndef BULKSNPE_HPP
#define BULKSNPE_HPP

#include <cstdlib>
#include "SNPE/SNPE.hpp"
#include "DlSystem/UserBufferMap.hpp"
#include "DlContainer/IDlContainer.hpp"
#include "DlSystem/DlEnums.hpp"
#include "UserBufferList.hpp"
#include "RuntimeConfigList.hpp"
#include "DlSystem/ZdlExportDefine.hpp"


/** @addtogroup c_plus_plus_apis C++
@{ */

/**
  * @brief .
  *
  * A structure snpe builder configuration
  */
struct ZDL_EXPORT BuildConfig final{
   zdl::DlContainer::IDlContainer* container;
   zdl::DlSystem::StringList outputBufferNames;
   RuntimeConfigList runtimeConfigList;
};

/**
  * @brief .
  *
  * The class for executing SNPE instances in parallel.
  */
class ZDL_EXPORT BulkSNPE {
public:

   ~BulkSNPE();

   /**
    * @brief Build snpe instance objects in parallel.
    *
    */
   bool build(BuildConfig &buildConfig);

   /**
    * @brief Execute snpe instance objects in parallel.
    *
    * @see zdl::SNPE
    */
   bool execute(const UserBufferList &inputBufferList, const UserBufferList &outputBufferList);

   /**
    * @brief Returns the input layer names of the network.
    *
    * @return StringList which contains the input layer names
    */
   const zdl::DlSystem::StringList getInputTensorNames() const noexcept;

   /**
    * @brief Returns the output layer names of the network.
    *
    * @return StringList which contains the output layer names
    */
   const zdl::DlSystem::StringList getOutputTensorNames() const noexcept;

   /**
    * @brief Returns the input tensor dimensions of the network.
    *
    * @return TensorShape which contains the dimensions.
    */
   const zdl::DlSystem::TensorShape getInputDimensions() const noexcept;

   /**
    * @brief Returns attributes of buffers.
    *
    * @see zdl::SNPE
    *
    * @return BufferAttributes of input/output tensor named.
    */
   const zdl::DlSystem::TensorShape getBufferAttributesDims(const char *name) const noexcept;

private:
   std::vector<std::unique_ptr<zdl::SNPE::SNPE>> m_snpeInstances;

};
#endif //BULKSNPE_HPP
