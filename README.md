![Zuko's banner](https://raw.githubusercontent.com/francois-rozet/zuko/master/docs/images/banner.svg)

# Zuko - Normalizing flows in PyTorch

Zuko is a Python package that implements normalizing flows in PyTorch. It relies as much as possible on distributions and transformations already provided by PyTorch. Unfortunately, the `Distribution` and `Transform` classes of `torch` are not sub-classes of `torch.nn.Module`, which means you cannot send their internal tensors to GPU with `.to('cuda')` or retrieve their parameters with `.parameters()`.

To solve this problem, `zuko` defines two abstract classes: `DistributionModule` and `TransformModule`. The former is any `Module` whose forward pass returns a `Distribution` and the latter is any `Module` whose forward pass returns a `Transform`. A normalizing flow is just a `DistributionModule` which contains a list of `TransformModule` and a base `DistributionModule`. This design allows for flows that behave like distributions while retaining the benefits of `Module`. It also makes the implementations easier to understand and extend.

> In the [Avatar](https://wikipedia.org/wiki/Avatar:_The_Last_Airbender) cartoon, [Zuko](https://wikipedia.org/wiki/Zuko) is a powerful firebender 🔥

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

Normalizing flows are provided in the `zuko.flows` module. To build one, supply the number of sample and context features as well as the transformations' hyperparameters. Then, feeding a context $c$ to the flow returns a conditional distribution $p(x | c)$ which can be evaluated and sampled from.

```python
import torch
import zuko

# Neural spline flow (NSF) with 3 sample features and 5 context features
flow = zuko.flows.NSF(3, 5, transforms=3, hidden_features=[128] * 3)

# Train to maximize the log-likelihood
optimizer = torch.optim.AdamW(flow.parameters(), lr=1e-3)

for x, c in trainset:
    loss = -flow(c).log_prob(x)  # -log p(x | c)
    loss = loss.mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Sample 64 points x ~ p(x | c*)
x = flow(c_star).sample((64,))
```

Alternatively, flows can be built as custom `FlowModule` objects.

```python
from zuko.flows import FlowModule, MaskedAutoregressiveTransform, Unconditional
from zuko.distributions import DiagNormal
from zuko.transforms import PermutationTransform

flow = FlowModule(
    transforms=[
        MaskedAutoregressiveTransform(3, 5, hidden_features=[128] * 3),
        Unconditional(PermutationTransform, torch.randperm(3), buffer=True),
        MaskedAutoregressiveTransform(3, 5, hidden_features=[128] * 3),
    ],
    base=Unconditional(
        DiagNormal,
        torch.zeros(3),
        torch.ones(3),
        buffer=True,
    ),
)
```

For more information, check out the documentation at [zuko.readthedocs.io](https://zuko.readthedocs.io).

### Available flows

| Class   | Year | Reference |
|:-------:|:----:|-----------|
| `MAF`   | 2017 | [Masked Autoregressive Flow for Density Estimation](https://arxiv.org/abs/1705.07057) |
| `NSF`   | 2019 | [Neural Spline Flows](https://arxiv.org/abs/1906.04032) |
| `NCSF`  | 2020 | [Normalizing Flows on Tori and Spheres](https://arxiv.org/abs/2002.02428) |
| `SOSPF` | 2019 | [Sum-of-Squares Polynomial Flow](https://arxiv.org/abs/1905.02325) |
| `NAF`   | 2018 | [Neural Autoregressive Flows](https://arxiv.org/abs/1804.00779) |
| `UNAF`  | 2019 | [Unconstrained Monotonic Neural Networks](https://arxiv.org/abs/1908.05164) |
| `CNF`   | 2018 | [Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366) |

## Contributing

If you have a question, an issue or would like to contribute, please read our [contributing guidelines](https://github.com/francois-rozet/zuko/blob/master/CONTRIBUTING.md).
