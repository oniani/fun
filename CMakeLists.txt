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

option(BUILD "Enable building" ON)
if(BUILD)
  message(STATUS "Enabling building...")
  add_executable(${PROJECT_NAME} main.cpp)
endif()

option(TESTING "Enable testing" OFF)
if(TESTING)
  message(STATUS "Enabling testing...")
  add_subdirectory(test)
endif()
