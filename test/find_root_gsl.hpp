#pragma once

#include "find_root.hpp"

namespace optimize {
namespace root {

/// find minimum of function f
Result find_root_gsl(const Fn& fn, const Vec& init, const Pred& stop_crit, const Config& config = Config{});

} // namespace root
} // namespace optimize
