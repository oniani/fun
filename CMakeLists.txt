cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(
  fun
  VERSION 0.0.1
  DESCRIPTION "A zero-dependency header-only activation function library in C++"
  HOMEPAGE_URL "https://github.com/oniani/fun"
  LANGUAGES C CXX
)

option(BUILD_EXECUTABLE "Enable building an executable" ON)
if(BUILD_EXECUTABLE)
  message(STATUS "Building an executable...")
  add_executable(${PROJECT_NAME} main.cpp)
endif()

option(TESTING "Enable testing" OFF)
if(ENABLE_TESTING)
  enable_testing()
  message("Building tests...")
  add_subdirectory(test)
endif()

if(NOT DEFINED COMPILER)
  set(COMPILER "clang")
endif()

if(${COMPILER} STREQUAL "clang")
  message(STATUS "Using clang...")
  include("cmake/toolchains/clang.cmake")
elseif(${COMPILER} STREQUAL "gcc")
  message(STATUS "Using gcc...")
  include("cmake/toolchains/gcc.cmake")
else()
  message(FATAL_ERROR "Unknown compiler")
endif()

include("cmake/tooling.cmake")
