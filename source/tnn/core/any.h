// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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

#ifndef TNN_INCLUDE_TNN_CORE_ANY_H_
#define TNN_INCLUDE_TNN_CORE_ANY_H_

#include <initializer_list>
#include <typeinfo>
#include <type_traits>
#include <utility>

#include "tnn/core/macro.h"

namespace TNN_NS {

// C++11 Does not support Variable Template, so we make a small change on in_place_type,
// Inplace Constructor of TNN::Any is a little bit different from that of std::any
// e.g.:
//struct A {
//    A(int age, std::string name, double salary) {}
//    int age;
//    std::string name;
//    double salary;
//};
//Any a2(any_in_place_type_t<A>{}, 30, std::string("Ada"), 1000.25);  // C++17 Style
//Any a2(any_in_place_type<A>, 30, std::string("Ada"), 1000.25);  // Our TNN implementation.


// ambiguation tags like std::inplace_type_t passed to the constructors of TNN::Any to indicate that the contained object should be constructed in-place, andthe type of the object to be constructed.
// C++17 Style
//template <class T>
//struct any_in_place_type_t {
//    explicit any_in_place_type_t() = default;
//};
//template <class T>
//constexpr any_in_place_type_t<T> any_in_place_type{};

// C++11 Style
template<class T>
struct any_in_place_type_tag {};
struct any_in_place_t {};
template<class T>
inline any_in_place_t any_in_place_type(any_in_place_type_tag<T> = any_in_place_type_tag<T>()) {
    return any_in_place_t();
}
#define any_in_place_type_t(T) any_in_place_t(&)(any_in_place_type_tag<T>)
#define any_in_place_type(T) any_in_place_type<T>



// @brief Simplified TNN::Any Class similar with C++17 std::any.
//class PUBLIC Any final {
class Any final {
public:
    //@brief Constructors of TNN::Any
    constexpr Any() noexcept : content_(nullptr) {}
    Any(const Any& other);
    Any(Any&& other) noexcept;

    template<class ValueType, class T = typename std::decay<ValueType>::type, typename std::enable_if<!std::is_same<T,Any>::value, int>::type = 0>
    Any(ValueType&& value) : content_(new TypedContent<T>(std::forward<ValueType>(value))) {}

    template<class ValueType, class... Args, typename std::enable_if<std::is_constructible<ValueType, Args&&...>::value, int>::type = 0>
    explicit Any(any_in_place_type_t(ValueType), Args&&... args) : content_(new TypedContent<ValueType>(ValueType(std::forward<Args>(args)...))) {}
    //explicit Any(any_in_place_type_t<ValueType>, Args&&... args) : content_(new TypedContent<ValueType>(ValueType(std::forward<Args>(args)...))) {}  // C++17 Style

    template<class ValueType, class U, class... Args, typename std::enable_if<std::is_constructible<ValueType, std::initializer_list<U>&, Args&&...>::value, int>::type = 0>
    explicit Any(any_in_place_type_t(ValueType), std::initializer_list<U> il, Args&&... args) : content_(new TypedContent<ValueType>(ValueType(il, std::forward<Args>(args)...))) {}
    //explicit Any(any_in_place_type_t<ValueType>, std::initializer_list<U> il, Args&&... args) : content_(new TypedContent<ValueType>(ValueType(il, std::forward<Args>(args)...))) {}  // C++17 Style
    
    //@brief Assigns contents to TNN::Any.
    Any& operator=(const Any& rhs);
    Any& operator=(Any&& rhs) noexcept;

    template<class ValueType, class T = typename std::decay<ValueType>::type, typename std::enable_if<!std::is_same<T,Any>::value, int>::type = 0>
    Any& operator=(T && value) {
        Any(std::move(value)).swap(*this);
        return *this;
    }

    //@brief Destructor of TNN::Any.
    ~Any();

    //@brief Modifiers of TNN::Any.
    // First destroys the current contained object (if any) by reset(), then constructs an object as the contained object.
    template<class ValueType, class... Args, typename std::enable_if<std::is_constructible<typename std::decay<ValueType>::type, Args&&...>::value, int>::type = 0>
    typename std::decay<ValueType>::type& emplace(Args&&... args) {
        Any(typename std::decay<ValueType>::type(std::forward<Args>(args)...)).swap(*this);
        return *internal_pointer_cast<typename std::decay<ValueType>::type>();
    }

    template<class ValueType, class U, class... Args, typename std::enable_if<std::is_constructible<typename std::decay<ValueType>::type, std::initializer_list<U>&, Args&&...>::value, int>::type = 0>
    typename std::decay<ValueType>::type& emplace(std::initializer_list<U> il, Args&&... args) {
        Any(typename std::decay<ValueType>::type(il, std::forward<Args>(args)...)).swap(*this);
        return *internal_pointer_cast<typename std::decay<ValueType>::type>();
    }

    // If not empty, destroys the contained object.
    void reset() noexcept;

    // Swaps the content of two any objects.
    void swap(Any& other) noexcept;

    //@brief Observers of TNN::Any.
    // Checks whether the object contains a value.
    bool has_value() const noexcept;

    // Queries the contained type.
    const std::type_info& type() const noexcept;

private:
    template<class T>
    friend T* any_cast(Any* operand) noexcept;
    
    template<class T>
    friend T* any_cast(Any* operand) noexcept;

    class Content {
    public:
        virtual ~Content() {}
        virtual std::type_info const& type() const = 0;
        virtual Content* clone() const = 0;
    };
    
    template<typename ValueType>
    class TypedContent : public Content {
    public:
        TypedContent(ValueType const& value) : value_(value) {}
        TypedContent(ValueType && value): value_(std::move(value)) {}
        virtual std::type_info const& type() const override {
            return typeid(ValueType);
        }
        virtual TypedContent* clone() const override {
            return new TypedContent(value_);
        }
        ValueType value_;
    };

    template<class ValueType>
    const ValueType* internal_pointer_cast() const {
        return &(static_cast<TypedContent<ValueType>*>(content_)->value_);
    }

    template<class ValueType>
    ValueType* internal_pointer_cast() {
        return &(static_cast<TypedContent<ValueType>*>(content_)->value_);
    }

    Content* content_;
};


// Non-member functions of TNN::Any
// Overloads the std::swap algorithm for std::any. Swaps the content of two any objects by calling lhs.swap(rhs).
void swap(Any& lhs, Any& rhs) noexcept;

// Defines a type of object to be thrown by the value-returning forms of std::any_cast on failure.
class bad_any_cast : public std::bad_cast {
public:
    virtual const char* what() const throw() {
        return "Bad TNN::Any Cast.";
    }
};

// Performs type-safe access to the contained object.
template<class T, typename = typename std::enable_if<(std::is_reference<T>::value || std::is_copy_constructible<T>::value), T>::type>
inline T any_cast(const Any& operand) {
    auto* ret = any_cast<typename std::remove_cv<typename  std::remove_reference<T>::type>::type>(&operand);
    //assert(ret); // If No Throw
    if (!ret) throw bad_any_cast();
    return static_cast<T>(*ret);
}

template<class T, typename = typename std::enable_if<(std::is_reference<T>::value || std::is_copy_constructible<T>::value), T>::type>
inline T any_cast(Any& operand) {
    auto* ret = any_cast<typename std::remove_cv<typename std::remove_reference<T>::type>::type>(&operand);
    //assert(ret); // If No Throw
    if (!ret) throw bad_any_cast();
    return static_cast<T>(*ret);
}

template<class T, typename = typename std::enable_if<(std::is_reference<T>::value || std::is_copy_constructible<T>::value), T>::type>
inline T any_cast(Any&& operand) {
    auto* ret = any_cast<typename std::remove_cv<typename  std::remove_reference<T>::type>::type>(&operand);
    //assert(ret); // If No Throw
    if (!ret) throw bad_any_cast();
    return static_cast<T>(std::move(*ret));
}

template<class T>
const T* any_cast(const Any* operand) noexcept {
    return operand != nullptr && operand->type() == typeid(T) ? operand->internal_pointer_cast<T>() : nullptr;
}

template<class T>
T* any_cast(Any* operand) noexcept {
    return operand != nullptr && operand->type() == typeid(T) ? operand->internal_pointer_cast<T>() : nullptr;
}

// Constructs an any object containing an object of type T, passing the provided arguments to T's constructor.
template<class T, class... Args>
Any make_any(Args&&... args) {
    //return Any(any_in_place_type<T>, std::forward<Args>(args)...);  // C++17 Style
    return Any(any_in_place_type(T), std::forward<Args>(args)...);
}

template<class T, class U, class... Args>
Any make_any(std::initializer_list<U> il, Args&&... args) {
    //return Any(any_in_place_type<T>, il, std::forward<Args>(args)...);  // C++17 Style
    return Any(any_in_place_type(T), il, std::forward<Args>(args)...);
}
}  // namespace TNN_NS

#endif  // TNN_INCLUDE_TNN_CORE_BLOB_IMPL_H_
