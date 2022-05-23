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

#ifndef CONSTEXPR_OPS_HPP
#define CONSTEXPR_OPS_HPP

namespace constexpr_ops {

/**
 * @brief Mathematical constant π.
 */
static const constexpr auto PI = 3.14159265358979323846264338327950288419716939937510;

/**
 * @brief Computes the power of the given number.
 * @param x Input value.
 * @param exp Exponent.
 * @return The value obtained by raising the input value to the given exponent.
 */
[[nodiscard]] constexpr auto pow(const double x, unsigned int exp) noexcept {
    auto res = x;
    while (--exp != 0) {
        res *= x;
    }
    return res;
}

/**
 * @brief Computes the factorial of an integer.
 * @param x Input value.
 * @return Factorial of the input value.
 */
[[nodiscard]] constexpr auto factorial(unsigned int num) noexcept {
    auto res = num;
    while (--num != 0) {
        res *= num;
    }
    return res;
}

/**
 * @brief Uses the Taylor series expansion to compute the value of the exp function.
 * @param x Input value.
 * @return exp of the input value.
 */
[[nodiscard]] constexpr auto exp(const double x) noexcept {
    return 1 + x + pow(x, 2) / factorial(2) + pow(x, 3) / factorial(3) + pow(x, 4) / factorial(4) +
           pow(x, 5) / factorial(5) + pow(x, 6) / factorial(6) + pow(x, 7) / factorial(7) +
           pow(x, 8) / factorial(8) + pow(x, 9) / factorial(9) + pow(x, 10) / factorial(10) +
           pow(x, 11) / factorial(11) + pow(x, 12) / factorial(12);
}

/**
 * @brief Uses the Taylor series expansion to compute the value of the sin function.
 * @param x Input value.
 * @return sin of the input value.
 */
[[nodiscard]] constexpr auto sin(const double x) noexcept {
    return x - pow(x, 3) / factorial(3) + pow(x, 5) / factorial(5) - pow(x, 7) / factorial(7) +
           pow(x, 9) / factorial(9) - pow(x, 11) / factorial(11) + pow(x, 13) / factorial(13);
}

/**
 * @brief Uses the Taylor series expansion to compute the value of the cos function.
 * @param x Input value.
 * @return cos of the input value.
 */
[[nodiscard]] constexpr auto cos(const double x) noexcept {
    return 1 - pow(x, 2) / factorial(2) + pow(x, 4) / factorial(4) - pow(x, 6) / factorial(6) +
           pow(x, 8) / factorial(8) - pow(x, 10) / factorial(10) + pow(x, 12) / factorial(12);
}

/**
 * @brief Uses the Taylor series expansion for computing the value of the cosh function.
 * @param x Input value.
 * @return cosh of the input value.
 */
[[nodiscard]] constexpr auto cosh(const double x) noexcept {
    return (exp(x) + exp(-x)) / 2;
}

/**
 * @brief Uses the Taylor series expansion for computing the value of the tanh function.
 * @param x Input value.
 * @return tanh of the input value.
 */
[[nodiscard]] constexpr auto tanh(const double x) noexcept {
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
}

/**
 * @brief Computes the square root of the given number using Newton's method.
 * @param x Input value.
 * @return The square root of the input value.
 */
[[nodiscard]] constexpr auto sqrt(const double x, const unsigned int max_iter = 15) noexcept {
    double res = x;
    for (unsigned int it = 0; it < max_iter; ++it) {
        res = 0.5 * (res + x / res);
    }
    return res;
}

/**
 * @brief Computes the natural logarithm of the given number via Halley-Newton method.
 * @param x Input value.
 * @param epsilon Parameter.
 * @return The natural logarithm of the input value.
 */
[[nodiscard]] constexpr auto ln(double x, double epsilon = 1e-5) noexcept {
    double y = x - 1.0;
    double z = y;
    auto abs = [](const auto a) { return a < 0 ? -a : a; };

    do {
        y = z;
        z = y + 2 * (x - exp(y)) / (x + exp(y));
    } while (abs(y - z) > epsilon);

    return z;
}

}  // namespace constexpr_ops

#endif  // CONSTEXPR_OPS_HPP
