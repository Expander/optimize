find_package(GSL REQUIRED)
add_definitions(-DENABLE_GSL=1)

# build google benchmark (target: benchmark)
# do not build tests of benchmarking lib
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Suppressing benchmark's tests" FORCE)
add_subdirectory(google/benchmark)

# build tests (targets: gtest_main, gtest)
add_subdirectory(google/googletest)

function(add_optimize_test name)
  add_executable(
    ${name}
    ${name}.cpp
    )
  target_link_libraries(
    ${name}
    optimize
    GSL::gsl
    ${ARGN}
    )
  add_test(NAME ${name} COMMAND ${name})
  target_link_libraries(${name} gtest benchmark::benchmark)
endfunction()

add_optimize_test(test_find_minimum)
