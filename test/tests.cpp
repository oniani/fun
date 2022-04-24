#include <catch2/catch.hpp>

#include "../include/fun.hpp"

#define RANGE_START 1
#define RANGE_END 10
#define STEP_SIZE 0.25

using f32 = float;

TEST_CASE("ReLU", "[relu]") {
  REQUIRE(fun::relu(static_cast<f32>(0)) == 0);
  for (f32 val = RANGE_START; val <= RANGE_END; val += STEP_SIZE) {
    REQUIRE(fun::relu(val) == val);
    REQUIRE(fun::relu(-val) == 0);
  }
}

TEST_CASE("Leaky ReLU", "[leakyrelu]") {
  REQUIRE(fun::leaky_relu(static_cast<f32>(0)) == 0);
  for (f32 val = RANGE_START; val <= RANGE_END; val += STEP_SIZE) {
    REQUIRE(fun::leaky_relu(val) == val);
    REQUIRE(fun::leaky_relu(-val) == 1e-2 * -val);
  }
}

TEST_CASE("Parametric ReLU", "[parametricrelu]") {
  std::array<f32, 4> params = {0.25, 0.5, 0.75, 1};

  for (f32 val = RANGE_START; val <= RANGE_END; val += STEP_SIZE) {
    REQUIRE(fun::parametric_relu(val, 0) == val);
    for (const auto& param : params) {
      REQUIRE(fun::parametric_relu(val, param) == val);
      REQUIRE(fun::parametric_relu(-val, param) == param * -val);
    }
  }
}
