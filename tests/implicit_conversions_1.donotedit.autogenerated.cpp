
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
        const auto len = (-1);
        const unsigned int x = len; static_assert(ceto::is_non_aggregate_init_and_if_convertible_then_non_narrowing_v<decltype(len), std::remove_cvref_t<decltype(x)>>);
    }

