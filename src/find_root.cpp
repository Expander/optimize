#include "find_root.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <Eigen/QR>

namespace optimize {
namespace root {

using Mat = Eigen::MatrixXd;

namespace {

constexpr Scalar deriv_eps = 1e-4;
constexpr Scalar max_step = 100.0;

Scalar max_abs(const Vec& x)
{
   return x.cwiseAbs().maxCoeff();
}

Scalar calc_max_dx(const Vec& x1, const Vec& x2)
{
   return (x1 - x2).cwiseAbs().cwiseProduct(x1.cwiseAbs().cwiseMax(1.0).cwiseInverse()).maxCoeff();
}

Scalar calc_max_step(const Vec& x)
{
   const Scalar sum = x.dot(x);
   const Scalar n = x.size();
   return max_step*std::max(std::sqrt(sum), n);
}

#define MSG(x) std::cout << x << std::endl;

Scalar calc_fmin(const Vec& x)
{
   return 0.5*x.dot(x);
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

Result find_root(Fn f, const Vec& init, Pred stop_crit, unsigned max_iter)
{
   Result res{init, f(init), 0, false};

   if (max_abs(res.y) < 0.01*deriv_eps)
      return res;

   const auto n = init.size();
   const Scalar stpmax = calc_max_step(res.x);
   const Scalar fmin = calc_fmin(res.y);
   Mat jac(n,n);
   Vec grad(n), xold(n), p(n);
   auto fold = fmin;

   while (res.iterations++ < max_iter && !res.found) {
      MSG("[" << res.iterations << "]: x = " << res.x.transpose() << ", f(x) = " << res.y.transpose());
      jac = fdjac(f, res.x, res.y);
      // compute grad(f) for line search
      grad = jac*res.y;
      // store x and fmin
      xold = res.x;
      fold = fmin;
      // solve linear equations by LU decomposition
      p = jac.colPivHouseholderQr().solve(-res.y);
      // do line search, @todo
      bool err = false;
      // scale dx
      // auto sum = std::sqrt(dx.dot(dx));
      // if (sum > stpmax)
      //    dx *= stpmax/sum;
      // do step
      MSG("doing step by dx = " << p.transpose());
      res.x = xold + p;
      res.y = f(res.x);
      // check for convergence
      res.found = stop_crit(res.y, calc_max_dx(res.x, xold));
      // check for convergence on function values
      if (res.found) {
         MSG("converged!");
         return res;
      }
      // check for grad(f) being zero (spurious convergence)
      if (err) {
         const Scalar den = std::max(fmin, 0.5*n);
         const Scalar max_grad = grad.cwiseAbs().cwiseProduct(res.x.cwiseAbs().cwiseMax(1.0)).maxCoeff()/den;
         res.found = max_grad >= 1e-6;
         return res;
      }
   }

   return res;
}

} // namespace root
} // namespace optimize
