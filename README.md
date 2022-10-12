![Zuko's banner](https://raw.githubusercontent.com/francois-rozet/zuko/master/sphinx/images/banner.svg)

# Zuko - Normalizing flows in PyTorch

Zuko is a Python package that implements normalizing flows in PyTorch. It relies as much as possible on distributions and transformations already provided by PyTorch. Unfortunately, the `Distribution` and `Transform` classes of `torch` are not sub-classes of `torch.nn.Module`, which means you cannot send their internal tensors to GPU with `.to('cuda')` or retrieve their parameters with `.parameters()`.

To solve this problem, `zuko` defines two abstract classes: `DistributionModule` and `TransformModule`. The former is any `Module` whose forward pass returns a `Distribution` and the latter is any `Module` whose forward pass returns a `Transform`. Then, a normalizing flow is the composition of a list of `TransformModule` and a base `DistributionModule`. This design allows for flows that behave like distributions while retaining the benefits of `Module`. It also makes the implementations easy to understand and extend.

> In the [Avatar](https://wikipedia.org/wiki/Avatar:_The_Last_Airbender) cartoon, Zuko is a powerful firebender 🔥

## Installation

The `zuko` package is available on [PyPI](https://pypi.org/project/zuko), which means it is installable via `pip`.

```
pip install zuko
```

Alternatively, if you need the latest features, you can install it from the repository.

```
pip install git+https://github.com/francois-rozet/zuko
```

## Getting started

Normalizing flows are provided in the `zuko.flows` module. To build one, supply the number of sample and context features as well as the transformations' hyperparameters. Then, feeding a context `y` to the flow returns a conditional distribution `p(x | y)` which can be evaluated and sampled from.

```python
import torch
import zuko

x = torch.randn(3)
y = torch.randn(5)

# Neural spline flow (NSF) with 3 transformations
flow = zuko.flows.NSF(3, 5, transforms=3, hidden_features=[128] * 3)

# Evaluate log p(x | y)
log_p = flow(y).log_prob(x)

# Sample 64 points x ~ p(x | y)
x = flow(y).sample((64,))
```

For more information about the available features check out the documentation at [francois-rozet.github.io/zuko](https://francois-rozet.github.io/zuko).

## Contributing

If you have a question, an issue or would like to contribute, please read our [contributing guidelines](CONTRIBUTING.md).
