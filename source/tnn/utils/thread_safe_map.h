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

class RWLock
{
public:
    inline void lock_read() {
        reading_threads_++;
        if(writing_threads_>0) {
            reading_threads_--;
            std::unique_lock<std::mutex> lock(read_mutex_);
            cond_.notify_all();
            while(writing_threads_>0) cond_.wait(lock);
            reading_threads_++;
        }
    }

    inline void unlock_read() {
        reading_threads_--;
        cond_.notify_all();
    }

    inline void lock_write() {
        writing_threads_++;
        std::unique_lock<std::mutex> lock(read_mutex_);
        while(reading_threads_>0) cond_.wait(lock);
        write_mutex_.lock();
    }

    inline void unlock_write() {
        writing_threads_--;
        cond_.notify_all();
        write_mutex_.unlock();
    }

private:
    std::atomic_int reading_threads_{0};
    std::atomic_int writing_threads_{0};
    std::condition_variable cond_;
    std::mutex read_mutex_;
    std::mutex write_mutex_;

};


template < class Key, class T, class Compare  =  std::less<Key>, class Alloc  =  std::allocator<std::pair<const Key,T> > > 
class thread_safe_map : public std::map<Key, T, Compare, Alloc>
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
    
    mutable RWLock rwlock_;
    
public:
    explicit thread_safe_map (const key_compare& comp  =  key_compare(), 
                              const allocator_type& alloc  =  allocator_type()) : std::map<Key, T, Compare, Alloc>(comp,alloc) {}

    explicit thread_safe_map (const allocator_type& alloc) : std::map<Key, T, Compare, Alloc>(alloc) {}

    template <class InputIterator> 
    thread_safe_map (InputIterator first, InputIterator last,
                    const key_compare& comp  =  key_compare(), const allocator_type& alloc  =  allocator_type()) : std::map<Key, T, Compare, Alloc>
                    (first, last, comp, alloc) {}

    thread_safe_map (const map& x) : std::map<Key, T, Compare, Alloc>(x) {}
    thread_safe_map (const thread_safe_map& x) : std::map<Key, T, Compare, Alloc>(x) {}
    thread_safe_map (const thread_safe_map& x, const allocator_type& alloc) : std::map<Key, T, Compare, Alloc>(x, alloc) {}
    thread_safe_map (map&& x) : std::map<Key, T, Compare, Alloc>(x) {}
    thread_safe_map (thread_safe_map&& x) : std::map<Key, T, Compare, Alloc>(x) {}
    thread_safe_map (map&& x, const allocator_type& alloc) : std::map<Key, T, Compare, Alloc>(x, alloc) {}
    thread_safe_map (thread_safe_map&& x, const allocator_type& alloc) : std::map<Key, T, Compare, Alloc>(x, alloc) {}

    thread_safe_map (std::initializer_list<value_type> il, const key_compare& comp  =  key_compare(), 
                     const allocator_type& alloc  =  allocator_type()) : std::map<Key, T, Compare, Alloc>(il, comp, alloc) {}

    thread_safe_map<Key, T, Compare, Alloc>& operator =  (const map& x) {
        rwlock_.lock_write();
        auto ret = std::map<Key, T, Compare, Alloc>::operator = (x);
        rwlock_.unlock_write();
        return (ret);
    }

    thread_safe_map<Key, T, Compare, Alloc>& operator =  (const thread_safe_map<Key, T, Compare, Alloc>& x) {
        rwlock_.lock_write();
        auto ret = std::map<Key, T, Compare, Alloc>::operator = (x);
        rwlock_.unlock_write();
        return (ret);
    }

    thread_safe_map<Key, T, Compare, Alloc>& operator =  (map&& x) {
        rwlock_.lock_write();
        auto ret = std::map<Key, T, Compare, Alloc>::operator = (x);
        rwlock_.unlock_write();
        return (ret);
    }

    thread_safe_map<Key, T, Compare, Alloc>& operator =  (thread_safe_map<Key, T, Compare, Alloc>&& x) {
        rwlock_.lock_write();
        auto ret = std::map<Key, T, Compare, Alloc>::operator = (x);
        rwlock_.unlock_write();
        return (ret);
    }

    thread_safe_map<Key, T, Compare, Alloc>& operator =  (std::initializer_list<value_type> il) {
        rwlock_.lock_write();
        auto ret = std::map<Key, T, Compare, Alloc>::operator = (il);
        rwlock_.unlock_write();
        return (ret);
    }

    bool empty() const noexcept {
        rwlock_.lock_read();
        auto ret = std::map<Key, T, Compare, Alloc>::empty();
        rwlock_.unlock_read();
        return (ret);
    }
     
    size_type size() const noexcept {
        rwlock_.lock_read();
        auto ret = std::map<Key, T, Compare, Alloc>::size();
        rwlock_.unlock_read();
        return (ret);
    }

    mapped_type& operator[] (const key_type& k) {
        rwlock_.lock_write();
        std::shared_ptr<void> defer(nullptr,[&](void* xx) {
            rwlock_.unlock_write();
        }); // deleter
        return std::map<Key, T, Compare, Alloc>::operator[](k);
    }

    mapped_type& operator[] (key_type&& k) {
        rwlock_.lock_write();
        std::shared_ptr<void> defer(nullptr,[&](void* xx) {
            rwlock_.unlock_write();
        }); // deleter
        return std::map<Key, T, Compare, Alloc>::operator[](k);
    }

    mapped_type& at (const key_type& k) {
        rwlock_.lock_write();
        std::shared_ptr<void> defer(nullptr,[&](void* xx) {
            rwlock_.unlock_write();
        }); // deleter
        
        return std::map<Key, T, Compare, Alloc>::at(k);
    }
    
    const mapped_type& at (const key_type& k) const {
        rwlock_.lock_write();
        std::shared_ptr<void> defer(nullptr,[&](void* xx) {
            rwlock_.unlock_write();
        }); // deleter
        
        return std::map<Key, T, Compare, Alloc>::at(k);
    }

    std::pair<iterator,bool> insert (const value_type& val) {
        rwlock_.lock_write();
        auto ret = std::map<Key, T, Compare, Alloc>::insert(val);
        rwlock_.unlock_write();
        return (ret);
    }

    std::pair<iterator,bool> insert (value_type&& val) {
        rwlock_.lock_write();
        auto ret = std::map<Key, T, Compare, Alloc>::insert(std::move(val));
        rwlock_.unlock_write();
        return (ret);
    }

    iterator insert (iterator position, const value_type& val) {
        rwlock_.lock_write();
        auto ret = std::map<Key, T, Compare, Alloc>::insert(position, val);
        rwlock_.unlock_write();
        return (ret);
    }

    iterator insert (iterator position, value_type&& val) {
        rwlock_.lock_write();
        auto ret = std::map<Key, T, Compare, Alloc>::insert(position, std::move(val));
        rwlock_.unlock_write();
        return (ret);
    }

    template <class InputIterator> void insert (InputIterator first, InputIterator last) {
        rwlock_.lock_write();
        auto ret = std::map<Key, T, Compare, Alloc>::insert(first, last);
        rwlock_.unlock_write();
        return (ret);
    }

    iterator erase (const_iterator position) {
        rwlock_.lock_write();
        auto ret = std::map<Key, T, Compare, Alloc>::erase(position);
        rwlock_.unlock_write();
        return (ret);
    }

    size_type erase (const key_type& k) {
        rwlock_.lock_write();
        auto ret = std::map<Key, T, Compare, Alloc>::erase(k);
        rwlock_.unlock_write();
        return (ret);
    }

    iterator erase (const_iterator first, const_iterator last) {
        rwlock_.lock_write();
        auto ret = std::map<Key, T, Compare, Alloc>::erase(first, last);
        rwlock_.unlock_write();
        return (ret);
    }

    void swap(map& x) {
        rwlock_.lock_write();
        std::map<Key, T, Compare, Alloc>::swap(x);
        rwlock_.unlock_write();
    }

    void swap(thread_safe_map<Key, T, Compare, Alloc>& x) {
        rwlock_.lock_write();
        x.lock_write();
        std::map<Key, T, Compare, Alloc>::swap(x);
        x.unlock_write();
        rwlock_.unlock_write();
    }

    void clear() noexcept {
        rwlock_.lock_write();
        std::map<Key, T, Compare, Alloc>::clear();
        rwlock_.unlock_write();
    }

    template <class... Args> std::pair<iterator,bool> emplace (Args&&... args) {
        rwlock_.lock_write();
        auto ret = std::map<Key, T, Compare, Alloc>::emplace(args...);
        rwlock_.unlock_write();
        return (ret);
    }

    template <class... Args> iterator emplace_hint (const_iterator position, Args&&... args) {
        rwlock_.lock_write();
        auto ret = std::map<Key, T, Compare, Alloc>::emplace_hint(position, args...);
        rwlock_.unlock_write();
        return (ret);
    }

    iterator find (const key_type& k) {
        rwlock_.lock_read();
        auto ret = std::map<Key, T, Compare, Alloc>::find(k);
        rwlock_.unlock_read();
        return (ret);
    }

    const_iterator find (const key_type& k) const {
        rwlock_.lock_read();
        auto ret = std::map<Key, T, Compare, Alloc>::find(k);
        rwlock_.unlock_read();
        return (ret);
    }

    size_type count (const key_type& k) const {
        rwlock_.lock_read();
        auto ret = std::map<Key, T, Compare, Alloc>::count(k);
        rwlock_.unlock_read();
        return (ret);
    }

    inline void lock_read() {(rwlock_.lock_read)();}
    inline void unlock_read() {(rwlock_.unlock_read)();}
    inline void lock_write() {(rwlock_.lock_write)();}
    inline void unlock_write() {(rwlock_.unlock_write)();}
};

} // namespace TNN_NS

#endif //TNN_SOURCE_UTILS_THREAD_SAFE_MAP_H_