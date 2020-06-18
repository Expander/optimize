#pragma once

#include <Eigen/Core>

namespace optimize {
namespace root {

using Vec = Eigen::VectorXd;
using Fn = std::function<Vec(const Vec&)>;
using Pred = std::function<bool(const Vec&)>;

struct Result {
   Vec x;             ///< point where zero is located
   bool found{false}; ///< zero has been found
};

/// find minimum of function f
Result find_root(Fn f, const Vec& init, Pred stop_crit);

} // namespace root
} // namespace optimize
