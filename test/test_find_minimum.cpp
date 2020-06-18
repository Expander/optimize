#include "gtest/gtest.h"
#include "find_minimum.hpp"
#include <cmath>

namespace {

constexpr double sqr(double x) noexcept { return x*x; }

} // anonymous namespace


TEST(test_parabola, test_1d)
{
   using namespace optimize::minimize;

   const double precision = 1e-10;
   const double xoffset = 0.0;
   const double yoffset = 0.0;

   const Fn f = [&xoffset, &yoffset] (const Vec& v) -> Scalar {
      return sqr(v(0) - xoffset) + yoffset;
   };

   const Pred stop_crit = [&precision] (const Vec& v1, const Vec& v2) -> bool {
      return std::abs(v1(0) - v2(0)) < precision;
   };

   Vec init(1);
   init << 2.0;

   const auto result = find_minimum(f, init, stop_crit);

   ASSERT_EQ(result.found, true);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
