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

#ifndef TNN_SOURCE_UTILS_THREAD_SAFE_MAP_H_
#define TNN_SOURCE_UTILS_THREAD_SAFE_MAP_H_

#include <memory>
#include <map>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <atomic>

#include "tnn/core/common.h"
#include "tnn/core/macro.h"

namespace TNN_NS {


template < class Key, class T, class Compare  =  std::less<Key>, class Alloc  =  std::allocator<std::pair<const Key,T> > > 
class safe_map : public std::map<Key, T, Compare, Alloc>
{
private:
    typedef typename std::map<Key, T, Compare, Alloc>::iterator iterator;
    typedef typename std::map<Key, T, Compare, Alloc>::const_iterator const_iterator;
    typedef typename std::map<Key, T, Compare, Alloc>::key_type key_type;
    typedef typename std::map<Key, T, Compare, Alloc>::key_compare key_compare;
    typedef typename std::map<Key, T, Compare, Alloc>::allocator_type allocator_type;
    typedef typename std::map<Key, T, Compare, Alloc>::value_type value_type;
    typedef typename std::map<Key, T, Compare, Alloc>::mapped_type mapped_type;
    typedef typename std::map<Key, T, Compare, Alloc>::size_type size_type;
    typedef typename std::map<Key, T, Compare, Alloc> map;
    
public:
    explicit safe_map (const key_compare& comp  =  key_compare(), 
                              const allocator_type& alloc  =  allocator_type()) : std::map<Key, T, Compare, Alloc>(comp,alloc) {}

    explicit safe_map (const allocator_type& alloc) : std::map<Key, T, Compare, Alloc>(alloc) {}

    template <class InputIterator> 
    safe_map (InputIterator first, InputIterator last,
                    const key_compare& comp  =  key_compare(), const allocator_type& alloc  =  allocator_type()) : std::map<Key, T, Compare, Alloc>
                    (first, last, comp, alloc) {}

    safe_map (const map& x) : std::map<Key, T, Compare, Alloc>(x) {}
    safe_map (const safe_map& x) : std::map<Key, T, Compare, Alloc>(x) {}
    safe_map (const safe_map& x, const allocator_type& alloc) : std::map<Key, T, Compare, Alloc>(x, alloc) {}
    safe_map (map&& x) : std::map<Key, T, Compare, Alloc>(x) {}
    safe_map (safe_map&& x) : std::map<Key, T, Compare, Alloc>(x) {}
    safe_map (map&& x, const allocator_type& alloc) : std::map<Key, T, Compare, Alloc>(x, alloc) {}
    safe_map (safe_map&& x, const allocator_type& alloc) : std::map<Key, T, Compare, Alloc>(x, alloc) {}

    safe_map (std::initializer_list<value_type> il, const key_compare& comp  =  key_compare(), 
                     const allocator_type& alloc  =  allocator_type()) : std::map<Key, T, Compare, Alloc>(il, comp, alloc) {}

    safe_map<Key, T, Compare, Alloc>& operator =  (const map& x) {
        std::map<Key, T, Compare, Alloc>::operator = (x);
        return (*this);
    }

    safe_map<Key, T, Compare, Alloc>& operator =  (const safe_map<Key, T, Compare, Alloc>& x) {
        std::map<Key, T, Compare, Alloc>::operator = (x);
        return (*this);
    }

    safe_map<Key, T, Compare, Alloc>& operator =  (map&& x) {
        std::map<Key, T, Compare, Alloc>::operator = (x);
        return (*this);
    }

    safe_map<Key, T, Compare, Alloc>& operator =  (safe_map<Key, T, Compare, Alloc>&& x) {
        std::map<Key, T, Compare, Alloc>::operator = (x);
        return (*this);
    }

    safe_map<Key, T, Compare, Alloc>& operator =  (std::initializer_list<value_type> il) {
        std::map<Key, T, Compare, Alloc>::operator = (il);
        return (*this);
    }

    mapped_type& operator[] (const key_type& k) {
        return std::map<Key, T, Compare, Alloc>::operator[](k);
    }

    const mapped_type operator[] (const key_type& k) const {
        auto it = std::map<Key, T, Compare, Alloc>::find(k);
        if (it == std::map<Key, T, Compare, Alloc>::end()) {
            return mapped_type();
        }
        return it->second;
    }

    mapped_type& operator[] (key_type&& k) {
        return std::map<Key, T, Compare, Alloc>::operator[](k);
    }
};

} // namespace TNN_NS

#endif //TNN_SOURCE_UTILS_THREAD_SAFE_MAP_H_