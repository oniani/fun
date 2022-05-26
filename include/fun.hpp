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
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
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

#include "constexpr_ops.hpp"

namespace fun {

/**
 * @brief Sigmoid activation function.
 * @param z Input value.
 * @return Value after activation via Sigmoid.
 */
[[nodiscard]] constexpr auto sigmoid(const double z) noexcept {
    auto expval = constexpr_ops::exp(z);
    return z < 0 ? expval / (1 + expval) : 1 / (1 + constexpr_ops::exp(-z));
}

/**
 * @brief Softmax activation function.
 * @param z Input vector.
 * @return Value after activation via Softmax.
 */
template <typename T>
[[nodiscard]] constexpr auto softmax(const T& zs) noexcept {
    auto acc = static_cast<T>(0);
    auto expsum = std::accumulate(zs.begin(), zs.end(), acc, [](const auto& lhs, const auto& rhs) {
        return lhs + constexpr_ops::exp(rhs);
    });

    std::vector<T> result = zs;
    std::for_each(result.begin(), result.end(),
                  [&](auto& val) { val = constexpr_ops::exp(val) / expsum; });

    return result;
}

/**
 * @brief Rectified Linear Unit (ReLU) activation function.
 * @param z Input value.
 * @return Value after activation via ReLU.
 */
[[nodiscard]] constexpr auto relu(const double z) noexcept {
    return z < 0 ? 0 : z;
}

/**
 * @brief Leaky ReLU activation function.
 * @param z Input value.
 * @return Value after activation via Leaky ReLU.
 */
[[nodiscard]] constexpr auto leaky_relu(const double z) noexcept {
    return z < 0 ? 1e-2 * z : z;
}

/**
 * @brief Parametric ReLU activation function.
 * @param z Input value.
 * @param a Scaling parameter.
 * @return Value after activation via Parametric ReLU.
 */
[[nodiscard]] constexpr auto parametric_relu(const double z, const double a) noexcept {
    return z < 0 ? a * z : z;
}

/**
 * @brief Gaussian Error Linear Unit (GELU) activation function.
 * @param z Input value.
 * @return Value after activation via GELU.
 */
[[nodiscard]] constexpr auto gelu(const double z) noexcept {
    return 0.5 * z *
           (1 + tanh(constexpr_ops::sqrt(2 / constexpr_ops::PI) * (z + 0.044715 * z * z * z)));
}

/**
 * @brief Sigmoid Linear Unit (SiLU) activation function.
 * @param z Input value.
 * @return Value after activation via SiLU.
 */
[[nodiscard]] constexpr auto silu(const double z) noexcept {
    return z * sigmoid(z);
}

/**
 * @brief Exponential Linear Units (ELU) activation function.
 * @param z Input value.
 * @param a Scale parameter.
 * @return Value after activation via ELU.
 */
[[nodiscard]] constexpr auto elu(const double z, const double a) noexcept {
    return z < 0 ? a * (constexpr_ops::exp(z) - 1) : z;
}

/**
 * @brief Softplus activation function.
 * @param z Input value.
 * @return Value after activation via Softplus.
 */
[[nodiscard]] constexpr auto softplus(const double z) noexcept {
    return constexpr_ops::ln(1 + constexpr_ops::exp(z));
}

/**
 * @brief Mish activation function.
 * @param z Input value.
 * @return Value after activation via Mish.
 */
[[nodiscard]] constexpr auto mish(const double z) noexcept {
    return z * tanh(softplus(z));
}

/**
 * @brief Identity activation function.
 * @param z Input value.
 * @return Value after activation via id.
 */
[[nodiscard]] constexpr auto id(const double z) noexcept {
    return z;
}

/**
 * @brief Binary step activation function.
 * @param z Input value.
 * @return Value after activation via binary step.
 */
[[nodiscard]] constexpr auto binary_step(const double z) noexcept {
    return z < 0 ? 0 : 1;
}

/**
 * @brief tanh activation function.
 * @param z Input value.
 * @return Value after activation via tanh.
 */
[[nodiscard]] constexpr auto tanh(const double z) noexcept {
    return constexpr_ops::tanh(z);
}

/**
 * @brief Gaussian activation function.
 * @param z Input value.
 * @return Value after activation via gaussian.
 */
[[nodiscard]] constexpr auto gaussian(const double z) noexcept {
    return constexpr_ops::exp(-z * z);
}

/**
 * @brief Growing cosine unit.
 * @param z Input value.
 * @return Value after activation via growing cosine unit.
 */
[[nodiscard]] constexpr auto gcs(const double z) noexcept {
    return z * constexpr_ops::cos(z);
}

namespace derivative {

/**
 * @brief Derivative of the sigmoid activation function.
 * @param z Input value.
 * @return Derivative of the sigmoid.
 */
[[nodiscard]] constexpr auto sigmoid(const double z) noexcept {
    auto sigval = fun::sigmoid(z);
    return sigval * (1 - sigval);
}

/**
 * @brief Derivative of the ReLU activation function.
 * @param z Input value.
 * @return ReLU derivative.
 */
[[nodiscard]] constexpr auto relu(const double z) noexcept {
    return z < 0 ? 0 : 1;
}

/**
 * @brief Derivative of the Leaky ReLU activation function.
 * @param z Input value.
 * @return Leaky ReLU derivative.
 */
[[nodiscard]] constexpr auto leaky_relu(const double z) noexcept {
    return z < 0 ? 1e-2 : 1;
}

/**
 * @brief Derivative of the parametric ReLU activation function.
 * @param z Input value.
 * @param a Scale parameter.
 * @return Parametric ReLU derivative.
 */
[[nodiscard]] constexpr auto parametric_relu(const double z, const double a) noexcept {
    return z < 0 ? a : 1;
}

/**
 * @brief Derivative of the GELU activation function.
 * @param z Input value.
 * @return GELU derivative.
 */
[[nodiscard]] constexpr auto gelu(const double z) noexcept {
    auto cube = z * z * z;
    auto tmp = 0.0356774 * cube + 0.797885 * z;
    return 0.5 * tanh(tmp) + (0.0535161 * cube + 0.398942 * z) / constexpr_ops::cosh(tmp) + 0.5;
}

/**
 * @brief Derivative of the SiLU activation function.
 * @param z Input value.
 * @return SiLU derivative.
 */
[[nodiscard]] constexpr auto silu(const double z) noexcept {
    return fun::sigmoid(z) + z * fun::derivative::sigmoid(z);
}

/**
 * @brief Derivative of the ELU activation function.
 * @param z Input value.
 * @param a Scale parameter.
 * @return ELU derivative.
 */
[[nodiscard]] constexpr auto elu(const double z, const double a) noexcept {
    return z < 0 ? a * constexpr_ops::exp(z) : 1;
}

/**
 * @brief Derivative of the Softplus activation function.
 * @param z Input value.
 * @return Softplus derivative.
 */
[[nodiscard]] constexpr auto softplus(const double z) noexcept {
    return fun::sigmoid(z);
}

/**
 * @brief Derivative of the Mish activation function.
 * @param z Input value.
 * @return Mish derivative.
 */
[[nodiscard]] constexpr auto mish(const double z) noexcept {
    auto omega = constexpr_ops::exp(3 * z) + 4 * constexpr_ops::exp(2 * z) +
                 (4 * z + 6) * constexpr_ops::exp(z) + 4 * (z + 1);
    auto tmp = constexpr_ops::exp(z) + 1;
    auto delta = tmp * tmp + 1;
    return constexpr_ops::exp(z) * omega / (delta * delta);
}

/**
 * @brief Derivative of the Identity activation function.
 * @param z Input value.
 * @return Identity derivative.
 */
[[nodiscard]] constexpr auto id([[maybe_unused]] const double z) noexcept {
    return 1;
}

/**
 * @brief Derivative of the Binary Step activation function.
 * @param z Input value.
 * @return Binary Step derivative.
 */
[[nodiscard]] constexpr auto binary_step([[maybe_unused]] const double z) noexcept {
    return 0;
}

/**
 * @brief Derivative of the tanh activation function.
 * @param z Input value.
 * @return tanh derivative.
 */
[[nodiscard]] constexpr auto tanh(const double z) noexcept {
    auto val = fun::tanh(z);
    return 1 - val * val;
}

/**
 * @brief Derivative of the Gaussian activation function.
 * @param z Input value.
 * @return Gaussian derivative.
 */
[[nodiscard]] constexpr auto gaussian(const double z) noexcept {
    return -2 * z * constexpr_ops::exp(-z * z);
}

/**
 * @brief Derivative of the Growing Cosine Unit (GCS).
 * @param z Input value.
 * @return GCS derivative.
 */
[[nodiscard]] constexpr auto gcs(const double z) noexcept {
    return constexpr_ops::cos(z) - z * constexpr_ops::sin(z);
}

}  // namespace derivative

}  // namespace fun

#endif  // FUN_HPP
