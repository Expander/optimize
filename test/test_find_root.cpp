#include "gtest/gtest.h"
#include "find_root.hpp"
#include "find_root_gsl.hpp"
#include <cmath>
#include <limits>
#include <tuple>

namespace {

constexpr double sqr(double x) noexcept { return x*x; }

} // anonymous namespace


class test_1d :public ::testing::TestWithParam<std::tuple<optimize::root::Fn, Eigen::VectorXd, bool> > {
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

   const Vec init = std::get<1>(GetParam());

   const auto result = find_root(f, init);

   ASSERT_EQ(result.found, std::get<2>(GetParam()));
   EXPECT_LT(result.iterations, config.max_iterations);
   if (result.found) {
      EXPECT_NEAR(result.y(0), 0.0, precision);
   }
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
              [] { Eigen::VectorXd v(1); v << 2.0; return v; }(), // init
              true // found
           ),
           // inverse gauss
           std::make_tuple(
              [] (const optimize::root::Vec& v) -> optimize::root::Vec {
                 optimize::root::Vec y(v.size());
                 y(0) = -std::exp(-v(0)*v(0)) + 0.5;
                 return y;
              },
              [] { Eigen::VectorXd v(1); v << 4.0; return v; }(), // init
              true // found
           ),
           // gauss
           std::make_tuple(
              [] (const optimize::root::Vec& v) -> optimize::root::Vec {
                 optimize::root::Vec y(v.size());
                 y(0) = std::exp(-v(0)*v(0));
                 return y;
              },
              [] { Eigen::VectorXd v(1); v << 1.0; return v; }(), // init
              true // found
           ),
           // gauss, starting point has vanishing gradient
           std::make_tuple(
              [] (const optimize::root::Vec& v) -> optimize::root::Vec {
                 optimize::root::Vec y(v.size());
                 y(0) = std::exp(-v(0)*v(0)) - 0.5;
                 return y;
              },
              [] { Eigen::VectorXd v(1); v << 0.0; return v; }(), // init
              true // found
           ),
           // no root, gauss with cut-off (vanishing gradient)
           std::make_tuple(
              [] (const optimize::root::Vec& v) -> optimize::root::Vec {
                 optimize::root::Vec y(v.size());
                 y(0) = std::abs(v(0)) < 2.0 ? std::exp(-v(0)*v(0)) : std::exp(-2.0*2.0);
                 return y;
              },
              [] { Eigen::VectorXd v(1); v << 1.0; return v; }(), // init
              false // found
           ),
           // no root, shifted gauss
           std::make_tuple(
              [] (const optimize::root::Vec& v) -> optimize::root::Vec {
                 optimize::root::Vec y(v.size());
                 y(0) = std::exp(-v(0)*v(0)) + 1.0;
                 return y;
              },
              [] { Eigen::VectorXd v(1); v << 1.0; return v; }(), // init
              false // found
           ),
           // no root, invalid function
           std::make_tuple(
              [] (const optimize::root::Vec& v) -> optimize::root::Vec {
                 optimize::root::Vec y(v.size());
                 y(0) = v(0) > 1.0 ? v(0) : std::numeric_limits<double>::quiet_NaN();
                 return y;
              },
              [] { Eigen::VectorXd v(1); v << 4.0; return v; }(), // init
              false // found
           ),
           // start at zero
           std::make_tuple(
              [] (const optimize::root::Vec& v) -> optimize::root::Vec {
                 optimize::root::Vec y(v.size());
                 y(0) = sqr(v(0) - 0.0) + 0.0;
                 return y;
              },
              [] { Eigen::VectorXd v(1); v << 0.0; return v; }(), // init
              true // found
           )
        ));


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
