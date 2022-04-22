#include <algorithm>
#include <array>
#include <iostream>

#include "include/fun.hpp"

template <typename T>
constexpr void print_vec(std::vector<T> xs) {
  std::cout << "Softmax:             ";
  std::copy(xs.begin(), xs.end(), std::ostream_iterator<T>(std::cout, " "));
  std::cout << '\n';
}

template <typename T, std::size_t N>
constexpr void print_arr(std::array<T, N> xs) {
  std::cout << "Softmax:             ";
  std::copy(xs.begin(), xs.end(), std::ostream_iterator<T>(std::cout, " "));
  std::cout << '\n';
}

using T = double;

int main() {
  std::cout << "---------------------\n";

  std::cout << "Sigmoid:             " << fun::sigmoid(4.0) << '\n';

  print_vec(fun::softmax(std::vector<T>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}));
  print_arr(fun::softmax(std::array<T, 10>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}));

  std::cout << "ReLU:                " << fun::relu(4.0) << '\n';
  std::cout << "Leaky ReLU:          " << fun::leaky_relu(4.0) << '\n';
  std::cout << "Parametric ReLU:     " << fun::parametric_relu(4.0, 0.2) << '\n';
  std::cout << "GELU:                " << fun::gelu(4.0) << '\n';
  std::cout << "SiLU:                " << fun::silu(4.0) << '\n';
  std::cout << "ELU:                 " << fun::elu(1.2, 0.2) << '\n';
  std::cout << "Softplus:            " << fun::softplus(4.0) << '\n';
  std::cout << "Identity:            " << fun::id(4.0) << '\n';
  std::cout << "Binary Step:         " << fun::binary_step(4.0) << '\n';
  std::cout << "tanh:                " << fun::tanh(4.0) << '\n';
  std::cout << "Gaussian:            " << fun::gaussian(4.0) << '\n';
  std::cout << "Growing Cosine Unit: " << fun::gcs(4.0) << '\n';

  std::cout << "\n---------------------\n";

  std::cout << "ReLU Derivative:     " << fun::derivative::relu(4.0) << '\n';
  // std::cout << "Leaky ReLU:          " << fun::leaky_relu(4.0) << '\n';
  // std::cout << "Parametric ReLU:     " << fun::parametric_relu(4.0, 0.2) << '\n';
  // std::cout << "GELU:                " << fun::gelu(4.0) << '\n';
  // std::cout << "SiLU:                " << fun::silu(4.0) << '\n';
  // std::cout << "ELU:                 " << fun::elu(1.2, 0.2) << '\n';
  // std::cout << "Softplus:            " << fun::softplus(4.0) << '\n';
  // std::cout << "Identity:            " << fun::id(4.0) << '\n';
  // std::cout << "Binary Step:         " << fun::binary_step(4.0) << '\n';
  // std::cout << "tanh:                " << fun::tanh(4.0) << '\n';
  // std::cout << "Gaussian:            " << fun::gaussian(4.0) << '\n';
  // std::cout << "Growing Cosine Unit: " << fun::gcs(4.0) << '\n';
}
