cmake_minimum_required(VERSION 3.1)
project(optimize VERSION 0.1.0 LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 3.1 REQUIRED)

enable_testing()
add_subdirectory(src)
add_subdirectory(test)
