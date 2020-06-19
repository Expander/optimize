#pragma once

#include <Eigen/Core>

namespace optimize {
namespace root {

using Scalar = double;
using Vec = Eigen::VectorXd;
using Fn = std::function<Vec(const Vec&)>;
using Pred = std::function<bool(const Vec&, Scalar)>;

struct Config {
   Scalar derivative_eps{1.0e-4}; ///< epsilon for numerical derivative
   unsigned max_iterations{200};  ///< maximum number of iterations
};

struct Result {
   Vec x;                  ///< point where zero is located
   Vec y;                  ///< function value at x
   unsigned iterations{0}; ///< number of iterations
   bool found{false};      ///< zero has been found
};

/// find minimum of function f
Result find_root(Fn fn, const Vec& init, Pred stop_crit, const Config& config = Config{});

} // namespace root
} // namespace optimize
