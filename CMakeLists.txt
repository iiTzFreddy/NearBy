cmake_minimum_required(VERSION 3.16)
project(NearBy LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# The plan's fairness rule: final numbers come from an -O2 Release build.
if(MSVC)
  set(CMAKE_CXX_FLAGS_RELEASE "/O2")
  add_compile_options(/W4 /permissive-)
else()
  set(CMAKE_CXX_FLAGS_RELEASE "-O2")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Add src/quadtree.cpp here when it lands.
add_library(nearby_core STATIC
  src/geo.cpp
  src/loader.cpp
  src/brute.cpp
  src/bench.cpp
  src/kdtree.cpp
)
target_include_directories(nearby_core PUBLIC src)

add_executable(nearby src/main.cpp)
target_link_libraries(nearby PRIVATE nearby_core)

enable_testing()
add_executable(nearby_tests tests/test_main.cpp)
target_link_libraries(nearby_tests PRIVATE nearby_core)
add_test(NAME nearby_tests COMMAND nearby_tests)
