#include <benchmark/benchmark.h>
#include "find_root.hpp"
#include <cmath>


const optimize::root::Fn gauss = [](const optimize::root::Vec& v) {
   optimize::root::Vec y(v.size());
   for (Eigen::Index i = 0; i < v.size(); ++i)
      y(i) = -std::exp(-v(i)*v(i)) + 0.5;
   return y;
};


const optimize::root::Pred stop_crit = [] (const optimize::root::Vec& v, optimize::root::Scalar max_dx) {
   constexpr double precision = 1e-10;
   return v.cwiseAbs().sum() < precision || max_dx < precision;
};


static void BM_find_root_gauss(benchmark::State& state)
{
   optimize::root::Vec init(state.range(0));
   init.setConstant(4.0);

   for (auto _ : state) {
      benchmark::DoNotOptimize(optimize::root::find_root(gauss, init, stop_crit));
   }
}


BENCHMARK(BM_find_root_gauss)->RangeMultiplier(2)->Range(1, 1 << 12);


BENCHMARK_MAIN();
