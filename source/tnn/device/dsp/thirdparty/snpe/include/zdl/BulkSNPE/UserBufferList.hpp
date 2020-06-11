//==============================================================================
//
//  Copyright (c) 2019 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================
#ifndef BULKSNPE_USERBUFFERLIST_HPP
#define BULKSNPE_USERBUFFERLIST_HPP

#include "DlSystem/UserBufferMap.hpp"
#include "DlSystem/ZdlExportDefine.hpp"
#include <vector>

/** @addtogroup c_plus_plus_apis C++
@{ */
/**
* The class for creating a UserBufferMap container.
*
*/
class ZDL_EXPORT UserBufferList final
{
public:
   UserBufferList();
   UserBufferList(const size_t size);
   void push_back(const zdl::DlSystem::UserBufferMap &userBufferMap);
   zdl::DlSystem::UserBufferMap& operator[](const size_t index);
   UserBufferList& operator =(const UserBufferList &other);
   size_t size() const noexcept;
   size_t capacity() const noexcept;
   void clear() noexcept;
   ~UserBufferList() = default;

private:
   void swap(const UserBufferList &other);
   std::vector<zdl::DlSystem::UserBufferMap> m_userBufferMaps;

};
#endif //BULKSNPE_USERBUFFERLIST_HPP
