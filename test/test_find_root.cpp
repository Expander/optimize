#include "gtest/gtest.h"
#include "find_root.hpp"
#include <cmath>

namespace {

constexpr double sqr(double x) noexcept { return x*x; }

} // anonymous namespace


TEST(test_parabola, test_1d)
{
   using namespace optimize::root;

   const double precision = 1e-10;
   const double xoffset = 0.0;
   const double yoffset = 0.0;
   const double xroot = xoffset - std::sqrt(-yoffset);
   const Config config;
   unsigned ncalls = 0;

   const Fn f = [xoffset, yoffset, &ncalls] (const Vec& v) -> Vec {
      ncalls++;
      Vec y(v.size());
      y(0) = sqr(v(0) - xoffset) + yoffset;
      return y;
   };

   const Pred stop_crit = [precision] (const Vec& v, Scalar max_dx) -> bool {
      return std::abs(v(0)) < precision || max_dx < precision;
   };

   Vec init(1);
   init << 2.0;

   const auto result = find_root(f, init, stop_crit, config);

   ASSERT_EQ(result.found, true);
   EXPECT_NEAR(result.y(0), 0.0, precision);
   EXPECT_NEAR(result.x(0), xroot, 1e-5);
   EXPECT_LT(result.iterations, config.max_iterations);
   EXPECT_LE(ncalls, 1 + 3*result.iterations);
   std::cout << "number of function calls: " << ncalls << std::endl;
}


TEST(test_inv_gauss, test_1d)
{
   using namespace optimize::root;

   const double precision = 1e-10;
   const Config config;
   unsigned ncalls = 0;

   const Fn f = [&ncalls] (const Vec& v) -> Vec {
      ncalls++;
      Vec y(v.size());
      y(0) = -std::exp(-v(0)*v(0)) + 0.5;
      return y;
   };

   const Pred stop_crit = [precision] (const Vec& v, Scalar max_dx) -> bool {
      return std::abs(v(0)) < precision || max_dx < precision;
   };

   Vec init(1);
   init << 4.0;

   const auto result = find_root(f, init, stop_crit, config);

   ASSERT_EQ(result.found, true);
   EXPECT_NEAR(result.y(0), 0.0, precision);
   // EXPECT_NEAR(result.x(0), xroot, 1e-5);
   EXPECT_LT(result.iterations, config.max_iterations);
   EXPECT_LE(ncalls, 1 + 3*result.iterations);
   std::cout << "number of function calls: " << ncalls << std::endl;
}


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
