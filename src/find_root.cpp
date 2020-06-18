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

Scalar calc_max_rel(const Vec& x1, const Vec& x2)
{
   return x1.cwiseAbs().cwiseProduct(x2.cwiseAbs().cwiseMax(1.0)).maxCoeff();
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


bool line_search(const Vec& xold, Scalar fold, const Vec& grad, Vec& p,
                 Vec& x, Scalar& fmin, Scalar stpmax, Fn f)
{
   const auto n = x.size();

   // scale p if attempted step is too big
   {
      const auto sum = std::sqrt(p.dot(p));
      if (sum > stpmax)
         p *= stpmax/sum;
   }

   const Scalar test = calc_max_rel(p, xold);
   Scalar alamin = 1e-7/test;
   Scalar alam = 1.0;

   while (true) {
      x = xold + alam*p;
      fmin = calc_fmin(x);
      break;
   }

   return false;
}


Result find_root(Fn f, const Vec& init, Pred stop_crit, unsigned max_iter)
{
   Result res{init, f(init), 0, false};

   if (max_abs(res.y) < 0.01*deriv_eps)
      return res;

   const auto n = init.size();
   const Scalar stpmax = calc_max_step(res.x);
   Scalar fmin = calc_fmin(res.y);
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
      const bool err = line_search(xold, fold, grad, p, res.x, fmin, stpmax, f);
      // do step
      MSG("doing step by dx = " << p.transpose());
      // res.x = xold + p;
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
         const Scalar max_grad = calc_max_rel(grad, res.x)/den;
         res.found = max_grad >= 1e-6;
         return res;
      }
   }

   return res;
}

} // namespace root
} // namespace optimize
