#pragma once

#include <Eigen/Core>

namespace optimize {
namespace minimize {

using Val = double;
using Vec = Eigen::VectorXd;
using Fn = std::function<Val(const Vec&)>;
using Pred = std::function<bool(const Vec&, const Vec&)>;

struct Result {
   Vec x;             ///< point where minimum is located
   Val min{};         ///< minimum valu, f(x) = min
   bool found{false}; ///< minimum has been found
};

/// find minimum of function f
Result find_minimum(Fn f, const Vec& init, Pred stop_crit);

} // namespace minimize
} // namespace optimize
