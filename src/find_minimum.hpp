#pragma once

#include <Eigen/Core>

namespace optimize {
namespace minimize {

using Scalar = double;
using Vec = Eigen::VectorXd;
using Fn = std::function<double(const Vec&)>;

struct Result {
   Vec x;             ///< point where minimum is located
   Scalar min{};      ///< minimum valu, f(x) = min
   bool found{false}; ///< minimum has been found
};

/// find minimum of function f
Result find_minimum(Fn f, const Vec& init, double precision);

} // namespace minimize
} // namespace optimize
