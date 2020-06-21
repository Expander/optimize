#pragma once

#include <Eigen/Core>

namespace optimize {
namespace root {

using Scalar = double;
using Vec = Eigen::VectorXd;
using Fn = std::function<Vec(const Vec&)>;
using Pred = std::function<bool(const Vec&, const Vec&)>;

struct Config {
   Scalar derivative_eps{1.0e-4}; ///< epsilon for numerical derivative
   Scalar max_step{100.0};        ///< maximum step length
   unsigned max_iterations{200};  ///< maximum number of iterations
};

struct Result {
   Vec x;                  ///< the zero of the function
   Vec y;                  ///< function value at x
   unsigned iterations{0}; ///< number of performed iterations
   bool found{false};      ///< zero has been found
};

extern const Pred default_stop_crit; ///< default stop criterion

/// find minimum of function f
Result find_root(const Fn& fn, const Vec& init,
                 const Pred& stop_crit = default_stop_crit,
                 const Config& config = Config{});

} // namespace root
} // namespace optimize
