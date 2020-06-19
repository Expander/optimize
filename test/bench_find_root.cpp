#include <benchmark/benchmark.h>
#include "find_root.hpp"
#include <cmath>


static void BM_find_root_gauss(benchmark::State& state)
{
   using namespace optimize::root;

   const double precision = 1e-10;
   const Fn fn = [](const Vec&v) -> Vec {
      Vec y(v.size());
      y(0) = -std::exp(-v(0)*v(0)) + 0.5;
       return y;
   };
   const Pred stop_crit = [precision] (const Vec& v, Scalar max_dx) -> bool {
      return std::abs(v(0)) < precision || max_dx < precision;
   };
   Vec init(1);
   init << 4.0;

  for (auto _ : state) {
    benchmark::DoNotOptimize(find_root(fn, init, stop_crit));
  }
}


BENCHMARK(BM_find_root_gauss);


BENCHMARK_MAIN();
