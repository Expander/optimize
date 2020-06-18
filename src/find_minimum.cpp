#include "find_minimum.hpp"

namespace optimize {
namespace minimize {

using Scalar = double;

constexpr int max_iter = 200;
constexpr Scalar deriv_eps = 1e-4;

Result find_minimum(Fn f, const Vec& init, Pred stop_crit)
{
   const auto n = init.size();
   int it = 0;
   Val val = f(init);
   Result res;

   while (it++ < max_iter) {
      // @todo implement me
   }

   return res;
}

} // namespace minimize
} // namespace optimize
