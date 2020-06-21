#include "find_root_gsl.hpp"
#include <cmath>
#include <limits>
#include <gsl/gsl_multiroots.h>
#include <gsl/gsl_vector.h>

namespace optimize {
namespace root {
namespace {

bool is_finite(const gsl_vector* x)
{
   for (std::size_t i = 0; i < x->size; i++) {
      if (!std::isfinite(gsl_vector_get(x, i)))
         return false;
   }
   return true;
}

Vec to_vec(const gsl_vector* x)
{
   Vec v(x->size);
   for (std::size_t i = 0; i < x->size; i++)
      v(i) = gsl_vector_get(x, i);
   return v;
}

static int gsl_function(const gsl_vector* x, void* parameters, gsl_vector* f)
{
   if (!is_finite(x)) {
      gsl_vector_set_all(f, std::numeric_limits<double>::max());
      return GSL_EDOM;
   }

   Fn* fun = static_cast<Fn*>(parameters);
   int status = GSL_SUCCESS;
   Vec arg(to_vec(x));
   Vec result(x->size);

   result.setConstant(std::numeric_limits<double>::max());

   try {
      result = (*fun)(arg);
      status = result.allFinite() ? GSL_SUCCESS : GSL_EDOM;
   } catch (...) {
      status = GSL_EDOM;
   }

   // copy result -> f
   for (std::size_t i = 0; i < x->size; i++)
      gsl_vector_set(f, i, result(i));

   return status;
}

const gsl_multiroot_fsolver_type* get_solver_type(GSL_solver_type solver_type)
{
   const gsl_multiroot_fsolver_type* st = gsl_multiroot_fsolver_hybrid;

   switch (solver_type) {
      case GSL_solver_type::hybrid:  st = gsl_multiroot_fsolver_hybrid;  break;
      case GSL_solver_type::hybrids: st = gsl_multiroot_fsolver_hybrids; break;
      case GSL_solver_type::newton:  st = gsl_multiroot_fsolver_dnewton; break;
      case GSL_solver_type::broyden: st = gsl_multiroot_fsolver_broyden; break;
      default: break;
   }

   return st;
}

} // anonymous namespace

/// find minimum of function f (GSL wrapper)
Result find_root_gsl(const Fn& fn, const Vec& init, const Pred& stop_crit, const Config& config, GSL_solver_type solver_type)
{
   const std::size_t n = init.size();
   Result res{init, Vec::Zero(n), 0, false};
   Fn function(fn);
   int status;
   std::size_t iter = 0;
   void* parameters = &function;
   gsl_multiroot_function f = {gsl_function, n, parameters};
   gsl_vector* x = gsl_vector_alloc(n);

   for (std::size_t i = 0; i < n; ++i)
      gsl_vector_set(x, 0, init(i));

   gsl_multiroot_fsolver* solver
      = gsl_multiroot_fsolver_alloc(get_solver_type(solver_type), n);

   gsl_multiroot_fsolver_set(solver, &f, x);

   do {
      iter++;
      status = gsl_multiroot_fsolver_iterate(solver);

      if (status)   // check if solver is stuck
         break;

      status = stop_crit(to_vec(gsl_multiroot_fsolver_f(solver)),
                         to_vec(gsl_multiroot_fsolver_dx(solver)))
                  ? GSL_SUCCESS
                  : GSL_CONTINUE;

   } while (status == GSL_CONTINUE && iter < config.max_iterations);

   res.x = to_vec(solver->x);
   res.found = status == GSL_SUCCESS;

   gsl_vector_free(x);
   gsl_multiroot_fsolver_free(solver);

   return res;
}

} // namespace root
} // namespace optimize
