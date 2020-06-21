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

static int gsl_function(const gsl_vector* x, void* parameters, gsl_vector* f)
{
   if (!is_finite(x)) {
      gsl_vector_set_all(f, std::numeric_limits<double>::max());
      return GSL_EDOM;
   }

   Fn* fun = static_cast<Fn*>(parameters);
   int status = GSL_SUCCESS;
   Vec arg(x->size);
   Vec result(x->size);

   for (std::size_t i = 0; i < x->size; i++)
      arg(i) = gsl_vector_get(x, i);

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

const gsl_multiroot_fsolver_type* solver_type_to_gsl_pointer()
{
   return gsl_multiroot_fsolver_hybrid;
}

} // anonymous namespace

/// find minimum of function f (GSL wrapper)
Result find_root_gsl(const Fn& fn, const Vec& init, const Pred& stop_crit, const Config& config)
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
      = gsl_multiroot_fsolver_alloc(solver_type_to_gsl_pointer(), n);

   gsl_multiroot_fsolver_set(solver, &f, x);

   do {
      iter++;
      status = gsl_multiroot_fsolver_iterate(solver);

      if (status)   // check if solver is stuck
         break;

      status = gsl_multiroot_test_residual(solver->f, 1e-10);
   } while (status == GSL_CONTINUE && iter < config.max_iterations);

   for (std::size_t i = 0; i < n; ++i)
      res.x(i) = gsl_vector_get(solver->x, i);

   res.found = status == GSL_SUCCESS;

   gsl_vector_free(x);
   gsl_multiroot_fsolver_free(solver);

   return res;
}

} // namespace root
} // namespace optimize
