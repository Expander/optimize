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

/// returns true on error, false otherwise
template <class Fmin>
bool line_search(const Vec& xold, Scalar fold, const Vec& grad, Vec& p,
                 Vec& x, Scalar& fmin, Scalar stpmax, Fmin func)
{
   // scale p if attempted step is too big
   {
      const auto sum = std::sqrt(p.dot(p));
      if (sum > stpmax)
         p *= stpmax/sum;
   }

   const Scalar slope = grad.dot(p);
   const Scalar alf = 1e-4;
   const Scalar alamin = 1e-7/calc_max_rel(p, xold);
   Scalar alam = 1, alam2 = 0;
   Scalar tmplam = 0, f2 = 0;

   while (true) {
      x = xold + alam*p;
      fmin = func(x);

      if (alam < alamin) {
         x = xold;
         return true; // error
      } else if (fmin <= fold + alf*alam*slope) {
         return false; // ok
      } else {
         if (alam == 1) {
            // first time
            tmplam = -slope/(2*(fmin - fold - slope));
         } else {
            const Scalar rhs1 = fmin - fold - alam*slope;
            const Scalar rhs2 = f2 - fold - alam2*slope;
            const Scalar a = (rhs1/(alam*alam) - rhs2/(alam2*alam2))/(alam - alam2);
            const Scalar b = (-alam2*rhs1/(alam*alam) + alam*rhs2/(alam2*alam2))/(alam - alam2);

            if (a == 0) {
               tmplam = -slope/(2.0*b);
            } else {
               const Scalar disc = b*b - 3*a*slope;
               if (disc < 0)
                  tmplam = 0.5*alam;
               else if (b <= 0.0)
                  tmplam = (-b + std::sqrt(disc))/(3*a);
               else
                  tmplam = -slope/(b + std::sqrt(disc));
            }
            if (tmplam > 0.5*alam)
               tmplam = 0.5*alam;
         }
      }

      alam2 = alam;
      f2 = fmin;
      alam = std::max(tmplam, 0.1*alam);
   }

   return false; // ok
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
      const bool err = line_search(xold, fold, grad, p, res.x, fmin, stpmax, [] (const Vec& x) { return calc_fmin(x); });
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
