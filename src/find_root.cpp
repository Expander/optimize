#include "find_root.hpp"
#include <algorithm>
#include <cmath>
#include <Eigen/QR>

#ifdef ENABLE_VERBOSE
#include <iostream>
#define VERBOSE_MSG(x) std::cerr << x << std::endl;
#define ERROR_MSG(x) std::cerr << "Error: " << x << std::endl;
#else
#define VERBOSE_MSG(x)
#define ERROR_MSG(x)
#endif

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

Scalar calc_max_quotient(const Vec& x1, const Vec& x2)
{
   return x1.cwiseAbs().cwiseProduct(x2.cwiseAbs().cwiseMax(1.0)).maxCoeff();
}

Scalar calc_norm(const Vec& x)
{
   return std::sqrt(x.dot(x));
}

Scalar calc_max_step(const Vec& x, Scalar max_step)
{
   const Scalar n = x.size();
   return max_step*std::max(calc_norm(x), n);
}

Scalar calc_fmin(const Vec& x)
{
   return 0.5*x.dot(x);
}

/// scale dx if attempted step is too big
void restrict_dx(Vec& dx, Scalar max_step)
{
   const Scalar norm = calc_norm(dx);
   if (norm > max_step)
      dx *= max_step/norm;
}

/// calculates Jacobian, y = f(x)
Mat fdjac(const Fn& f, const Vec& x, const Vec& y, Scalar derivative_eps)
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

Scalar calc_lam(Scalar fold, Scalar fmin, Scalar fmin2, Scalar lam, Scalar lam2, Scalar slope) noexcept
{
   Scalar new_lam = 0;

   if (lam == 1) {
      // first time
      new_lam = -slope/(2*(fmin - fold - slope));
   } else {
      const Scalar rhs1 = fmin - fold - lam*slope;
      const Scalar rhs2 = fmin2 - fold - lam2*slope;
      const Scalar a = (rhs1/(lam*lam) - rhs2/(lam2*lam2))/(lam - lam2);
      const Scalar b = (-lam2*rhs1/(lam*lam) + lam*rhs2/(lam2*lam2))/(lam - lam2);

      if (a == 0) {
         new_lam = -slope/(2*b);
      } else {
         const Scalar disc = b*b - 3*a*slope;
         if (disc < 0)
            new_lam = 0.5*lam;
         else if (b <= 0)
            new_lam = (-b + std::sqrt(disc))/(3*a);
         else
            new_lam = -slope/(b + std::sqrt(disc));
      }
      if (new_lam > 0.5*lam)
         new_lam = 0.5*lam;
   }

   return new_lam;
}

} // anonymous namespace

/// returns true on error, false otherwise
bool line_search(const Vec& xold, Scalar fold, const Vec& grad, const Vec& dx,
                 Vec& x, Vec& y, Scalar& fmin, const Fn& fn)
{
   constexpr bool ok = false, error = true;
   const Scalar slope = grad.dot(dx);
   const Scalar alpha = 1e-4;
   const Scalar min_lam = 1e-7/calc_max_quotient(dx, xold);
   Scalar lam = 1, lam2 = 0;
   Scalar fmin2 = 0;

   while (true) {
      x = xold + lam*dx;
      y = fn(x);
      fmin = calc_fmin(y);

      if (!std::isfinite(fmin))
         return error;

      if (lam < min_lam) {
         x = xold;
         fmin = fold;
         return error;
      }

      if (fmin <= fold + alpha*lam*slope)
         break;

      const Scalar new_lam = calc_lam(fold, fmin, fmin2, lam, lam2, slope);

      lam2 = lam;
      fmin2 = fmin;
      lam = std::max(new_lam, 0.1*lam);
   }

   return ok;
}


Result find_root(const Fn& fn, const Vec& init, const Pred& stop_crit, const Config& config)
{
   Result res{init, fn(init), 0, false};

   if (!init.allFinite() || !res.y.allFinite())
      return res;

   if (max_abs(res.y) < 0.01*config.derivative_eps) {
      res.found = true;
      return res;
   }

   const auto n = init.size();
   const Scalar max_step = calc_max_step(res.x, config.max_step);
   Scalar fmin = calc_fmin(res.y);
   Mat jac(n,n);
   Vec grad(n), xold(n), dx(n);
   Scalar fold = fmin;

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

      restrict_dx(dx, max_step);

      grad = jac*res.y;
      const bool err = line_search(xold, fold, grad, dx, res.x, res.y, fmin, fn);

      if (!res.x.allFinite() || !res.y.allFinite() || !std::isfinite(fmin))
         return res;

      res.found = stop_crit(res.y, calc_max_dx(res.x, xold));

      if (res.found)
         return res;

      // check for grad(f) being zero (spurious convergence)
      if (err) {
         const Scalar den = std::max(fmin, 0.5*n);
         const Scalar max_grad = calc_max_quotient(grad, res.x)/den;
         res.found = max_grad >= 1e-6;
         return res;
      }
   }

   return res;
}

} // namespace root
} // namespace optimize
