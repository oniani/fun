# fun

A zero-dependency header-only activation function library in C++.

## Supported Activation Functions

- Sigmoid and Related Activation Functions:

  - [Sigmoid][sigmoid]
  - [Softmax][softmax]

- ReLU and Related Activation Functions:

  - Linear Variants:

    - [ReLU][relu]
    - [Leaky ReLU][leakyrelu]
    - [Parametric ReLU][parametricrelu]

  - Non-linear Variants:

    - [GELU][gelu]
    - [SiLU][silu]
    - [ELU][elu]
    - [Softplus][softplus]
    - [Mish][mish]

- Other Activation Functions:

  - [Identity][identity]
  - [Binary Step][binarystep]
  - [tanh][tanh]
  - [Gaussian][gaussian]
  - [Growing Cosine Unit][growingcosineunit]

## API

```cpp
#include "fun.hpp"

int main() {
  auto z = softplus(1.2);
  std::cout << z << '\n';
}
```

## Build

```console
$ mkdir build
$ cd build
$ cmake -DCOMPILER=clang -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=VCPKG_TOOLCHAIN_FILE ..
$ cmake --build .
```

## References

- [Activation function][activationfunction]

## License

[MIT License][license]

[sigmoid]: https://en.wikipedia.org/wiki/Sigmoid_function
[softmax]: https://en.wikipedia.org/wiki/Softmax_function
[relu]: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
[leakyrelu]: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)#Leaky_ReLU
[parametricrelu]: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)#Parametric_ReLUQ
[gelu]: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)#Gaussian_Error_Linear_Unit_(GELU)
[silu]: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)#SiLU
[elu]: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)#ELU
[softplus]: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)#Softplus
[mish]: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)#Mish
[identity]: https://en.wikipedia.org/wiki/Identity_function
[binarystep]: https://en.wikipedia.org/wiki/Heaviside_step_function
[tanh]: https://en.wikipedia.org/wiki/Hyperbolic_functions#Hyperbolic_tangent
[gaussian]: https://en.wikipedia.org/wiki/Gaussian_function
[growingcosineunit]: https://arxiv.org/abs/2108.12943
[activationfunction]: https://en.wikipedia.org/wiki/Activation_function
[license]: LICENSE
