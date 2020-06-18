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

   const Fn f = [&xoffset, &yoffset] (const Vec& v) -> Vec {
      Vec y(v.size());
      y(0) = sqr(v(0) - xoffset) + yoffset;
      return y;
   };

   const Pred stop_crit = [&precision] (const Vec& v) -> bool {
      return std::abs(v(0)) < precision;
   };

   Vec init(1);
   init << 2.0;

   const auto result = find_root(f, init, stop_crit);

   const double xroot = xoffset - std::sqrt(-yoffset);

   ASSERT_EQ(result.found, true);
   EXPECT_NEAR(f(result.x)(0), 0.0, precision);
   EXPECT_NEAR(result.x(0), xroot, 1e-5);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
