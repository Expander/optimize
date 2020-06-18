#include "find_root.hpp"
#include <cmath>

namespace optimize {
namespace root {

using Mat = Eigen::MatrixXd;
using Scalar = double;

namespace {

constexpr Scalar deriv_eps = 1e-4;
constexpr Scalar max_step = 100.0;

Scalar fmin(Fn f, const Vec& x)
{
   const auto y = f(x);
   return 0.5*y.dot(y);
}

/// calculates Jacobian, y = f(x)
Mat fdjac(Fn f, const Vec& x, const Vec& y)
{
   const Eigen::Index n = x.size();
   Mat jac(n,n);
   Vec xn(x), yn(n);

   for (Eigen::Index i = 0; i < n; ++i) {
      const double temp = xn(i);
      double h = deriv_eps*std::abs(temp);
      if (h == 0)
         h = deriv_eps;
      xn(i) = temp + h;
      h = xn(i) - temp;
      yn = f(xn);
      xn(i) = temp;
      jac.row(i) = (yn - y)/h;
   }

   return jac;
}

} // anonymous namespace

constexpr int max_iter = 200;

Result find_root(Fn f, const Vec& init, Pred stop_crit)
{
   const auto n = init.size();
   int it = 0;
   Result res;

   while (it++ < max_iter && !res.found) {
      // @todo implement me
   }

   return res;
}

} // namespace root
} // namespace optimize
