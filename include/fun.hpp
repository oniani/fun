/*
 * MIT License
 *
 * Copyright (c) 2022 David Oniani
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef FUN_HPP
#define FUN_HPP

#include <array>
#include <cmath>
#include <numeric>
#include <vector>

namespace fun {

  // Concepts and types {{{

  /**
   * @brief Architecture-dependent size type used across the library.
   */
  using size_type = std::size_t;

  /**
   * @brief Concept representing an arithmetic value (i.e., a number).
   */
  template <typename T>
  concept arithmetic = std::is_arithmetic_v<T>;

  // }}}

  // Sigmoid and Related Activation Functions {{{

  /**
   * @brief Sigmoid activation function.
   * @param z Input value.
   * @return Value after activation via Sigmoid.
   */
  [[nodiscard]] constexpr auto sigmoid(const arithmetic auto z) noexcept {
    auto expval = std::exp(z);
    return z < 0 ? expval / (1 + expval) : 1 / (1 + std::exp(-z));
  }

  /**
   * @brief Softmax activation function.
   * @param z Input array.
   * @return Value after activation via Softmax.
   */
  template <arithmetic T, size_type N>
  [[nodiscard]] constexpr auto softmax(const std::array<T, N>& zs) noexcept {
    auto acc = static_cast<T>(0);
    auto expsum = std::accumulate(zs.begin(), zs.end(), acc, [](const auto& lhs, const auto& rhs) {
      return lhs + std::exp(rhs);
    });

    std::array<T, N> result = zs;
    std::for_each(result.begin(), result.end(), [&](auto& val) { val = std::exp(val) / expsum; });

    return result;
  }

  /**
   * @brief Softmax activation function.
   * @param z Input vector.
   * @return Value after activation via Softmax.
   */
  template <arithmetic T>
  [[nodiscard]] constexpr auto softmax(const std::vector<T>& zs) noexcept {
    auto acc = static_cast<T>(0);
    auto expsum = std::accumulate(zs.begin(), zs.end(), acc, [](const auto& lhs, const auto& rhs) {
      return lhs + std::exp(rhs);
    });

    std::vector<T> result = zs;
    std::for_each(result.begin(), result.end(), [&](auto& val) { val = std::exp(val) / expsum; });

    return result;
  }

  // }}}

  // ReLU and Related Activation Functions {{{

  /**
   * @brief Rectified Linear Unit (ReLU) activation function.
   * @param z Input value.
   * @return Value after activation via ReLU.
   */
  [[nodiscard]] consteval auto relu(const arithmetic auto z) noexcept { return z < 0 ? 0 : z; }

  /**
   * @brief Leaky ReLU activation function.
   * @param z Input value.
   * @return Value after activation via leaky ReLU.
   */
  [[nodiscard]] consteval auto leaky_relu(const arithmetic auto z) noexcept {
    return z < 0 ? 1e-2 * z : z;
  }

  /**
   * @brief Parametric ReLU activation function.
   * @param z Input value.
   * @param a Scaling parameter.
   * @return Value after activation via leaky ReLU.
   */
  [[nodiscard]] consteval auto parametric_relu(const arithmetic auto z,
                                               const arithmetic auto a) noexcept {
    return z < 0 ? a * z : z;
  }

  /**
   * @brief Gaussian Error Linear Unit (GELU) activation function.
   * @param z Input value.
   * @return Value after activation via GELU.
   */
  [[nodiscard]] constexpr auto gelu(const arithmetic auto z) noexcept {
    return 0.5 * z * (1 + tanh(0.797885 * z + 0.35677 * z * z * z));
  }

  /**
   * @brief Sigmoid Linear Unit (SiLU) activation function.
   * @param z Input value.
   * @return Value after activation via SiLU.
   */
  [[nodiscard]] constexpr auto silu(const arithmetic auto z) noexcept { return z * sigmoid(z); }

  /**
   * @brief Exponential Linear Units (ELU) activation function.
   * @param z Input value.
   * @param a Parameter.
   * @return Value after activation via ELU.
   */
  [[nodiscard]] constexpr auto elu(const arithmetic auto z, const arithmetic auto a) noexcept {
    // static_assert(!(a < 0));
    return z < 0 ? a * (std::exp(z) - 1) : z;
  }

  /**
   * @brief Softplus activation function.
   * @param z Input value.
   * @return Value after activation via Softplus.
   */
  [[nodiscard]] constexpr auto softplus(const arithmetic auto z) noexcept {
    return std::log(1 + std::exp(z));
  }

  /**
   * @brief Mish activation function.
   * @param z Input value.
   * @return Value after activation via Mish.
   */
  [[nodiscard]] constexpr auto mish(const arithmetic auto z) noexcept {
    return z * tanh(softplus(z));
  }

  // }}}

  // Other Activation Functions {{{

  /**
   * @brief Identity activation function.
   * @param z Input value.
   * @return Value after activation via id.
   */
  [[nodiscard]] consteval auto id(const arithmetic auto z) noexcept { return z; }

  /**
   * @brief Binary step activation function.
   * @param z Input value.
   * @return Value after activation via binary step.
   */
  [[nodiscard]] consteval auto binary_step(const arithmetic auto z) noexcept {
    return z < 0 ? 0 : 1;
  }

  /**
   * @brief tanh activation function.
   * @param z Input value.
   * @return Value after activation via tanh.
   */
  [[nodiscard]] constexpr auto tanh(const arithmetic auto z) noexcept { return std::tanh(z); }

  /**
   * @brief Gaussian activation function.
   * @param z Input value.
   * @return Value after activation via gaussian.
   */
  [[nodiscard]] constexpr auto gaussian(const arithmetic auto z) noexcept {
    return std::exp(-z * z);
  }

  /**
   * @brief Growing cosine unit.
   * @param z Input value.
   * @return Value after activation via growing cosine unit.
   */
  [[nodiscard]] constexpr auto gcs(const arithmetic auto z) noexcept { return z * std::cos(z); }

  // }}}

  // Derivatives {{{

  namespace derivative {

    // Sigmoid and Related Activation Functions {{{

    /**
     * @brief Derivative of the sigmoid activation function.
     * @param z Input value.
     * @return Derivative of the sigmoid.
     */
    [[nodiscard]] constexpr auto sigmoid(const arithmetic auto z) noexcept {
      auto sigval = fun::sigmoid(z);
      return sigval * (1 - sigval);
    }

    // }}}

    // ReLU and Related Activation Functions {{{

    /**
     * @brief Derivative of the ReLU activation function.
     * @param z Input value.
     * @return ReLU derivative.
     */
    [[nodiscard]] consteval auto relu(const arithmetic auto z) noexcept { return z < 0 ? 0 : 1; }

    /**
     * @brief Derivative of the Leaky ReLU activation function.
     * @param z Input value.
     * @return Leaky ReLU derivative.
     */
    [[nodiscard]] consteval auto leaky_relu(const arithmetic auto z) noexcept {
      return z < 0 ? 1e-2 : 1;
    }

    /**
     * @brief Derivative of the parametric ReLU activation function.
     * @param z Input value.
     * @param a Scaling parameter.
     * @return Parametric ReLU derivative.
     */
    [[nodiscard]] consteval auto parametric_relu(const arithmetic auto z,
                                                 const arithmetic auto a) noexcept {
      return z < 0 ? a : 1;
    }

    /** TODO
     * @brief Derivative of the GELU activation function.
     * @param z Input value.
     * @return GELU derivative.
     */
    [[nodiscard]] constexpr auto gelu(const arithmetic auto z) noexcept {
      return 0.5 * z * (1 + tanh(0.797885 * z + 0.35677 * z * z * z));
    }

    /**
     * @brief Derivative of the SiLU activation function.
     * @param z Input value.
     * @return SiLU derivative.
     */
    [[nodiscard]] constexpr auto silu(const arithmetic auto z) noexcept {
      return fun::sigmoid(z) + z * fun::derivative::sigmoid(z);
    }

    /**
     * @brief Derivative of the ELU activation function.
     * @param z Input value.
     * @param a Parameter.
     * @return ELU derivative.
     */
    [[nodiscard]] constexpr auto elu(const arithmetic auto z, const arithmetic auto a) noexcept {
      // static_assert(!(a < 0));
      return z < 0 ? a * std::exp(z) : 1;
    }

    /**
     * @brief Derivative of the Softplus activation function.
     * @param z Input value.
     * @return Softplus derivative.
     */
    [[nodiscard]] constexpr auto softplus(const arithmetic auto z) noexcept {
      return fun::sigmoid(z);
    }

    /** TODO
     * @brief Derivative of the Mish activation function.
     * @param z Input value.
     * @return Mish derivative.
     */
    [[nodiscard]] constexpr auto mish(const arithmetic auto z) noexcept {
      return z * tanh(softplus(z));
    }

    // }}}

    // Other Activation Functions {{{

    /**
     * @brief Derivative of the identity activation function.
     * @param z Input value.
     * @return Value after activation via id.
     */
    [[nodiscard]] consteval auto id(const arithmetic auto z) noexcept { return 1; }

    /**
     * @brief Derivative of the binary step activation function.
     * @param z Input value.
     * @return Value after activation via binary step.
     */
    [[nodiscard]] consteval auto binary_step(const arithmetic auto z) noexcept { return 0; }

    /**
     * @brief Derivative of the tanh activation function.
     * @param z Input value.
     * @return Value after activation via tanh.
     */
    [[nodiscard]] constexpr auto tanh(const arithmetic auto z) noexcept {
      auto val = fun::derivative::tanh(z);
      return 1 - val * val;
    }

    /**
     * @brief Derivative of the gaussian activation function.
     * @param z Input value.
     * @return Value after activation via gaussian.
     */
    [[nodiscard]] constexpr auto gaussian(const arithmetic auto z) noexcept {
      return -2 * z * std::exp(-z * z);
    }

    /**
     * @brief Derivative of the growing cosine unit.
     * @param z Input value.
     * @return Value after activation via gcs.
     */
    [[nodiscard]] constexpr auto gcs(const arithmetic auto z) noexcept {
      return std::cos(z) - z * std::sin(z);
    }

    // }}}

  }  // namespace derivative

  // }}}

}  // namespace fun

#endif  // FUN_HPP
