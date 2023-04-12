#ifndef CETO_H
#define CETO_H

#include <memory>
#include <vector>
#include <type_traits>
#include <utility>
#include <source_location>

namespace ceto {

struct object {
};

struct shared_object : public std::enable_shared_from_this<shared_object>, object {
};

// answer from https://stackoverflow.com/questions/657155/how-to-enable-shared-from-this-of-both-parent-and-derived/47789633#47789633
// (perhaps it's possible to use the accepted answer without freestanding funcs
//  however this solution works with template classes (naive insertion of:
//      const auto& self = std::static_pointer_cast<std::remove_reference<decltype((*this))>::type>(shared_from_this())
//  does not!)

template <typename Base>
inline std::shared_ptr<Base>
shared_from_base(std::enable_shared_from_this<Base>* base) {
    return base->shared_from_this();
}

template <typename Base>
inline std::shared_ptr<const Base>
shared_from_base(std::enable_shared_from_this<Base> const* base) {
    return base->shared_from_this();
}

template <typename That>
inline std::shared_ptr<That>
shared_from(That* that) {
    return std::static_pointer_cast<That>(shared_from_base(that));
}


// mad = maybe allow deref

class null_deref_error : public std::runtime_error
{
public:

    static inline std::string build_message(const std::source_location& location) {
        std::string message = "Attempted null deref in attribute access:";
        message += location.file_name();
        message += ":";
        message += std::to_string(location.line());
        message += " (" + std::string(location.function_name()) + ")";
        message += " column " + std::to_string(location.column()) + "\n";
        return message;
    }

    using std::runtime_error::runtime_error;
};

template<typename T>
T* mad(T & obj) {
    return &obj;  // no autoderef:
}

// e.g. string temporaries
template<typename T>
T* mad(T && obj) {
    return &obj;   // no autoderef
}

template<typename T>
std::enable_if_t<std::is_base_of_v<object, T>, std::shared_ptr<T>&>
mad(std::shared_ptr<T>& obj, const std::source_location& location = std::source_location::current()) {
    if (!obj) {
        throw null_deref_error(null_deref_error::build_message(location));
    }
    return obj;   // autoderef
}
// autoderef:
//template<typename T>
//std::enable_if_t<std::is_base_of_v<object, T>, std::shared_ptr<T>>
//mad(std::shared_ptr<T> obj) { return obj; }


// autoderef of temporary
template<typename T>
std::enable_if_t<std::is_base_of_v<object, T>, std::shared_ptr<T>>
mad(std::shared_ptr<T>&& obj, const std::source_location& location = std::source_location::current()) {
    if (!obj) {
        throw null_deref_error(null_deref_error::build_message(location));
    }
    return std::move(obj);  // autoderef
}
//template<typename T>
//std::enable_if_t<std::is_base_of_v<object, T>, std::shared_ptr<T>&>
//mad(std::shared_ptr<T>&& obj) { return obj; }  // autoderef

// autoderef
template<typename T>
std::enable_if_t<std::is_base_of_v<object, T>, const std::shared_ptr<T>&>
mad(const std::shared_ptr<T>& obj, const std::source_location& location = std::source_location::current()) {
    if (!obj) {
        throw null_deref_error(null_deref_error::build_message(location));
    }
    return obj;
}

// autoderef
template<typename T>
std::enable_if_t<std::is_base_of_v<object, T>, std::unique_ptr<T>&>
mad(std::unique_ptr<T>& obj, const std::source_location& location = std::source_location::current()) {
    if (!obj) {
        throw null_deref_error(null_deref_error::build_message(location));
    }
    return obj;
}

// autoderef
template<typename T>
std::enable_if_t<std::is_base_of_v<object, T>, const std::unique_ptr<T>&>
mad(const std::unique_ptr<T>& obj, const std::source_location& location = std::source_location::current()) {
    if (!obj) {
        throw null_deref_error(null_deref_error::build_message(location));
    }
    return obj;
}

// autoderef of temporary
template<typename T>
std::enable_if_t<std::is_base_of_v<object, T>, std::unique_ptr<T>>
mad(std::unique_ptr<T>&& obj, const std::source_location& location = std::source_location::current()) {
    if (!obj) {
        throw null_deref_error(null_deref_error::build_message(location));
    }
    return std::move(obj);
}
//template<typename T>
//std::enable_if_t<std::is_base_of_v<object, T>, std::unique_ptr<T>&>
//mad(std::unique_ptr<T>&& obj) { return obj; }


// automatic make_shared insertion

template<typename T, typename... Args>
std::enable_if_t<std::is_base_of_v<shared_object, T>, std::shared_ptr<T>>
call_or_construct(Args&&... args) {
    // use braced args to disable narrowing conversions
    using TT = decltype(T{std::forward<Args>(args)...});

    return std::make_shared<TT>(std::forward<Args>(args)...);
}

// no braced call for 0-args case - avoid needing to define an explicit no-arg constructor
template<typename T>
std::enable_if_t<std::is_base_of_v<shared_object, T>, std::shared_ptr<T>>
call_or_construct() {
    return std::make_shared<T>();
}

template<typename T, typename... Args>
std::enable_if_t<std::is_base_of_v<object, T> && !std::is_base_of_v<shared_object, T>, std::unique_ptr<T>>
call_or_construct(Args&&... args) {
    using TT = decltype(T{std::forward<Args>(args)...});
    return std::make_unique<TT>(std::forward<Args>(args)...);
}

template<typename T>
std::enable_if_t<std::is_base_of_v<object, T> && !std::is_base_of_v<shared_object, T>, std::unique_ptr<T>>
call_or_construct() {
    return std::make_unique<T>();
}

// non-object concrete classes/structs (in C++ sense)
template <typename T, typename... Args>
std::enable_if_t<!std::is_base_of_v<object, T> /*&& !std::is_void_v<T>*/, T>
call_or_construct(Args&&... args) {
    return T{std::forward<Args>(args)...};
}

template <typename T>
std::enable_if_t<!std::is_base_of_v<object, T> && !std::is_void_v<T>, T>
call_or_construct() {
    return T();
}

// non-type template param version needed for e.g. construct_or_call<printf>("hi")
template<auto T, typename... Args>
auto
call_or_construct(Args&&... args) {
    return T(std::forward<Args>(args)...);
}

// template classes (forwarding to call_or_construct again seems to handle both object derived and plain classes)
template<template<class ...> class T, class... TArgs>
auto
call_or_construct(TArgs&&... args) {
    using TT = decltype(T(std::forward<TArgs>(args)...));
    return call_or_construct<TT>(T(std::forward<TArgs>(args)...));
}


// preparation for auto-move last use of unique object

template<typename T, typename Enable = void>
struct is_object_unique_ptr {
    enum { value = false };
};

template<typename T>
struct is_object_unique_ptr<T, typename std::enable_if<std::is_same<typename std::remove_cv<T>::type, std::unique_ptr<typename T::element_type>>::value
                                                       && std::is_base_of_v<object, T>>::type>
{
   enum { value = true };
};

// this one may be controversial (strong capture of shared object references by default - use 'weak' to break cycle)
template <class T>
std::enable_if_t<std::is_base_of_v<object, T>, std::shared_ptr<T>>
constexpr default_capture(std::shared_ptr<T> t) noexcept
{
    return t;
}

template <class T>
std::enable_if_t<std::is_arithmetic_v<T> || std::is_enum_v<T>, T>
constexpr default_capture(T t) noexcept
{
    return t;
}

// https://open-std.org/JTC1/SC22/WG21/docs/papers/2020/p0870r2.html#P0608R3
// still not in c++20 below is_convertible_without_narrowing implementation taken from
// https://github.com/GHF/mays/blob/db8b6b5556cc465d326d9e1acdc5483c70999b18/mays/internal/type_traits.h // (C) Copyright 2020 Xo Wang <xo@geekshavefeelings.com> // SPDX-License-Identifier: Apache-2.0

// True if |From| is implicitly convertible to |To| without going through a narrowing conversion.
// Will likely be included in C++2b through WG21 P0870 (see
// https://github.com/cplusplus/papers/issues/724).
template <typename From, typename To, typename Enable = void>
struct is_convertible_without_narrowing : std::false_type {};

// Implement "construct array of From" technique from P0870R4 with SFINAE instead of requires.
template <typename From, typename To>
struct is_convertible_without_narrowing<
    From,
    To,
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    std::void_t<decltype(std::type_identity_t<To[]>{std::declval<From>()})>> : std::true_type {};

template <typename From, typename To>
constexpr bool is_convertible_without_narrowing_v =
    is_convertible_without_narrowing<From, To>::value;

static_assert(!is_convertible_without_narrowing_v<float, int>, "float -> int is narrowing!");


// for our own diy impl of "no narrowing conversions in local var definitions"
// that avoids some pitfalls always printing ceto "x: Type = y" as c++ "Type x {y}"
// e.g. ceto code "l : std.vector<int> = 1") should be an error not an aggregate
// initialization.
template <typename From, typename To>
inline constexpr bool is_non_aggregate_init_and_if_convertible_then_non_narrowing_v =
    std::is_aggregate_v<From> == std::is_aggregate_v<To> &&
    (!std::is_convertible_v<From, To> ||
     is_convertible_without_narrowing_v<From, To>);


// just let .at() do the bounds checking
auto maybe_bounds_check_access(auto&& v, auto&& index) -> decltype(auto)
    requires (std::is_integral_v<std::remove_cvref_t<decltype(index)>> &&
              requires { std::size(v); v.at(index); v[index]; } &&
              std::is_same_v<decltype(v.at(index)), decltype(v[index])>)
{
    return std::forward<decltype(v)>(v).at(std::forward<decltype(index)>(index));
}

auto maybe_bounds_check_access(auto&& v, auto&& index) -> decltype(auto)
    requires (!(std::is_integral_v<std::remove_cvref_t<decltype(index)>>))
{
    return std::forward<decltype(v)>(v)[std::forward<decltype(index)>(index)];
}

} // namespace ceto

#endif // CETO_H