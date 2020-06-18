#include "find_root.hpp"
#include <algorithm>
#include <cmath>

namespace optimize {
namespace root {

using Mat = Eigen::MatrixXd;
using Scalar = double;

namespace {

constexpr Scalar deriv_eps = 1e-4;
constexpr Scalar max_step = 100.0;
constexpr Scalar min_dx = 1e-7;

Scalar max_abs(const Vec& x)
{
   return x.cwiseAbs().maxCoeff();
}

Scalar step_max(const Vec& x)
{
   const Scalar sum = x.dot(x);
   const Scalar n = x.size();
   return max_step*std::max(std::sqrt(sum), n);
}

// Scalar fmin(Fn f, const Vec& x)
// {
//    const auto y = f(x);
//    return 0.5*y.dot(y);
// }

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
   Result res{init, false};
   Vec fvec = f(init);

   if (max_abs(fvec) < 0.01*deriv_eps)
      return res;

   const auto stpmax = step_max(res.x);
   const auto n = init.size();
   const auto fmin = 0.5*fvec.dot(fvec);
   int it = 0;
   Mat jac(n,n);
   Vec grad(n), xold(n), p(n);
   auto fold = fmin;

   while (it++ < max_iter && !res.found) {
      jac = fdjac(f, res.x, fvec);
      // compute grad(f) for line search
      grad = jac*fvec;
      // store x and fmin
      xold = res.x;
      fold = fmin;
      // r.h.s. or linear equations
      p = -fvec;
      // solve linear equations by LU decomposition
      // @todo
      // do line search
      // @todo
      // check for convergence on function values
      if (max_abs(fvec) < deriv_eps)
         return res;
      // check for grad(f) being zero (spurious convergence)
      {
         const auto den = std::max(fmin, 0.5*n);
         const auto max_grad = grad.cwiseAbs().cwiseProduct(res.x.cwiseAbs().cwiseMax(1.0))/den;
      }
      // check for convergence on dx
      {
         const auto max_dx = (res.x - xold).cwiseAbs().cwiseProduct(res.x.cwiseAbs().cwiseMax(1.0).cwiseInverse()).maxCoeff();
         if (max_dx < min_dx) {
            res.found = true;
            return res;
         }
      }
   }

   return res;
}

} // namespace root
} // namespace optimize
