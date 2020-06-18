#include "find_root.hpp"

namespace optimize {
namespace root {

using Scalar = double;

constexpr int max_iter = 200;
constexpr Scalar deriv_eps = 1e-4;

Result find_root(Fn f, const Vec& init, Pred stop_crit)
{
   const auto n = init.size();
   int it = 0;
   Result res;

   while (it++ < max_iter) {
      // @todo implement me
   }

   return res;
}

} // namespace root
} // namespace optimize
