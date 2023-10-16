
#include <string>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <functional>
#include <cassert>
#include <compare> // for <=>
#include <thread>
#include <optional>

//#include <concepts>
//#include <ranges>
//#include <numeric>


#include "ceto.h"

    auto main() -> int {
        const bool b { true } ; static_assert(std::is_convertible_v<decltype(true), decltype(b)>);
        const bool b2 { 1 } ; static_assert(std::is_convertible_v<decltype(1), decltype(b2)>);
        const int i { true } ; static_assert(std::is_convertible_v<decltype(true), decltype(i)>);
        const unsigned int u2 { 5 } ; static_assert(std::is_convertible_v<decltype(5), decltype(u2)>);
        const unsigned int & ur = u2; static_assert(ceto::is_non_aggregate_init_and_if_convertible_then_non_narrowing_v<decltype(u2), std::remove_cvref_t<decltype(ur)>>);
        const auto um = ur;
        static_assert(std::is_const_v<decltype(um)>);
        (((((std::cout << b) << b2) << i) << u2) << ur) << um;
    }

