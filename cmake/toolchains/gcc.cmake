include("${CMAKE_CURRENT_LIST_DIR}/base.cmake")

set(PROJECT_TEMPLATE_CXX_COMPILER_MINIMUM_VERSION 10)

set(CMAKE_C_COMPILER "gcc")
set(CMAKE_CXX_COMPILER "g++")

set(CMAKE_AR "ar")
set(CMAKE_RANLIB "ranlib")

string(
  JOIN " " CMAKE_C_FLAGS_RELEASE
  "${CMAKE_C_FLAGS_RELEASE}"
  -fuse-ld=lld
)
