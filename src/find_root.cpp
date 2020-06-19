#include "find_root.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <Eigen/QR>

#define VERBOSE_MSG(x) std::cerr << x << std::endl;
#define ERROR_MSG(x) std::cerr << "Error: " << x << std::endl;

namespace optimize {
namespace root {
namespace {

using Mat = Eigen::MatrixXd;

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
   constexpr Scalar max_step = 100.0;
   const Scalar sum = x.dot(x);
   const Scalar n = x.size();
   return max_step*std::max(std::sqrt(sum), n);
}

Scalar calc_fmin(const Vec& x)
{
   return 0.5*x.dot(x);
}

/// calculates Jacobian, y = f(x)
Mat fdjac(Fn f, const Vec& x, const Vec& y, Scalar derivative_eps)
{
   const Eigen::Index n = x.size();
   Mat jac(n,n);
   Vec xn(x), yn(n);

   for (Eigen::Index i = 0; i < n; ++i) {
      const double temp = xn(i);
      double h = derivative_eps*std::abs(temp);
      if (h == 0)
         h = derivative_eps;
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
bool line_search(const Vec& xold, Scalar fold, const Vec& grad, Vec& dx,
                 Vec& x, Scalar& fmin, Scalar max_step)
{
   constexpr bool ok = false, error = true;

   // scale dx if attempted step is too big
   {
      const auto sum = std::sqrt(dx.dot(dx));
      if (sum > max_step) {
         const double scale = max_step/sum;
         if (std::abs(scale) <= std::numeric_limits<double>::epsilon())
            return error;
         dx *= scale;
      }
   }

   const Scalar slope = grad.dot(dx);
   const Scalar alf = 1e-4;
   const Scalar alamin = 1e-7/calc_max_rel(dx, xold);
   Scalar alam = 1, alam2 = 0;
   Scalar tmplam = 0, fmin2 = 0;

   while (true) {
      x = xold + alam*dx;
      fmin = calc_fmin(x);
      VERBOSE_MSG("adjust x by dx = " << alam*dx.transpose() << " (fmin = " << fmin << ")");

      if (alam < alamin) {
         x = xold;
         return error; // error
      } else if (fmin <= fold + alf*alam*slope) {
         return ok;
      } else {
         if (alam == 1) {
            // first time
            tmplam = -slope/(2*(fmin - fold - slope));
         } else {
            const Scalar rhs1 = fmin - fold - alam*slope;
            const Scalar rhs2 = fmin2 - fold - alam2*slope;
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
      fmin2 = fmin;
      alam = std::max(tmplam, 0.1*alam);
   }

   return ok;
}


Result find_root(Fn fn, const Vec& init, Pred stop_crit, const Config& config)
{
   Result res{init, fn(init), 0, false};

   if (max_abs(res.y) < 0.01*config.derivative_eps)
      return res;

   const auto n = init.size();
   const Scalar max_step = calc_max_step(res.x);
   Scalar fmin = calc_fmin(res.y);
   Mat jac(n,n);
   Vec grad(n), xold(n), dx(n);
   auto fold = fmin;

   while (res.iterations++ < config.max_iterations && !res.found) {
      VERBOSE_MSG("[" << res.iterations << "]: x = " << res.x.transpose() << ", f(x) = " << res.y.transpose());

      xold = res.x;
      fold = fmin;

      jac = fdjac(fn, res.x, res.y, config.derivative_eps);
      dx = jac.colPivHouseholderQr().solve(-res.y);

      if (!dx.allFinite()) {
         ERROR_MSG("dx is infinite");
         return res;
      }

      grad = jac*res.y;
      const bool err = line_search(xold, fold, grad, dx, res.x, fmin, max_step);

      res.y = fn(res.x);
      res.found = stop_crit(res.y, calc_max_dx(res.x, xold));

      if (res.found)
         return res;

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
