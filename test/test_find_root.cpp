#include "gtest/gtest.h"
#include "find_root.hpp"
#include <cmath>
#include <tuple>

namespace {

constexpr double sqr(double x) noexcept { return x*x; }

} // anonymous namespace


class test_1d :public ::testing::TestWithParam<std::tuple<optimize::root::Fn, double> > {
};


TEST_P(test_1d, root)
{
   using namespace optimize::root;

   const double precision = 1e-10;
   const Config config;
   unsigned ncalls = 0;

   auto fn = std::get<0>(GetParam());

   const Fn f = [&ncalls, &fn] (const Vec& v) -> Vec {
      ncalls++;
      return fn(v);
   };

   const Pred stop_crit = [precision] (const Vec& v, Scalar max_dx) -> bool {
      return std::abs(v(0)) < precision || max_dx < precision;
   };

   Vec init(1);
   init << std::get<1>(GetParam());

   const auto result = find_root(f, init, stop_crit, config);

   ASSERT_EQ(result.found, true);
   EXPECT_NEAR(result.y(0), 0.0, precision);
   EXPECT_LT(result.iterations, config.max_iterations);
   std::cout << "number of function calls: " << ncalls << std::endl;
}


INSTANTIATE_TEST_SUITE_P(
        tests_1d,
        test_1d,
        ::testing::Values(
           // parabola
           std::make_tuple(
              [] (const optimize::root::Vec& v) -> optimize::root::Vec {
                 optimize::root::Vec y(v.size());
                 y(0) = sqr(v(0) - 0.0) + 0.0;
                 return y;
              },
              2.0
           ),
           // inverse gauss
           std::make_tuple(
              [] (const optimize::root::Vec& v) -> optimize::root::Vec {
                 optimize::root::Vec y(v.size());
                 y(0) = -std::exp(-v(0)*v(0)) + 0.5;
                 return y;
              },
              4.0
           )
        ));


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
