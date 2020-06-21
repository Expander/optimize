#pragma once

#include "find_root.hpp"

namespace optimize {
namespace root {

enum class GSL_solver_type {
   hybrid, hybrids, newton, broyden
};

/// find minimum of function f
Result find_root_gsl(const Fn& fn, const Vec& init,
                     const Pred& stop_crit = default_stop_crit,
                     const Config& config = Config{},
                     GSL_solver_type solver_type = GSL_solver_type::hybrid);

} // namespace root
} // namespace optimize
