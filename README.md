![Zuko's banner](https://raw.githubusercontent.com/probabilists/zuko/master/docs/images/banner.svg)

# Zuko - Normalizing flows in PyTorch

Zuko is a Python package that implements normalizing flows in [PyTorch](https://pytorch.org). It relies as much as possible on distributions and transformations already provided by PyTorch. Unfortunately, the `Distribution` and `Transform` classes of `torch` are not sub-classes of `torch.nn.Module`, which means you cannot send their internal tensors to GPU with `.to('cuda')` or retrieve their parameters with `.parameters()`. Worse, the concepts of conditional distribution and transformation, which are essential for probabilistic inference, are impossible to express.

To solve these problems, `zuko` defines two concepts: the `LazyDistribution` and `LazyTransform`, which are any modules whose forward pass returns a `Distribution` or `Transform`, respectively. Because the creation of the actual distribution/transformation is delayed, an eventual condition can be easily taken into account. This design enables lazy distributions, including normalizing flows, to act like distributions while retaining features inherent to modules, such as trainable parameters. It also makes the implementations easy to understand and extend.

> In the [Avatar](https://wikipedia.org/wiki/Avatar:_The_Last_Airbender) cartoon, [Zuko](https://wikipedia.org/wiki/Zuko) is a powerful firebender 🔥

## Acknowledgements

Zuko takes significant inspiration from [nflows](https://github.com/bayesiains/nflows) and [Stefan Webb](https://github.com/stefanwebb)'s work in [Pyro](https://github.com/pyro-ppl/pyro) and [FlowTorch](https://github.com/facebookincubator/flowtorch).

## Installation

The `zuko` package is available on [PyPI](https://pypi.org/project/zuko), which means it is installable via `pip`.

```
pip install zuko
```

Alternatively, if you need the latest features, you can install it from the repository.

```
pip install git+https://github.com/probabilists/zuko
```

## Getting started

Normalizing flows are provided in the `zuko.flows` module. To build one, supply the number of sample and context features as well as the transformations' hyperparameters. Then, feeding a context $c$ to the flow returns a conditional distribution $p(x | c)$ which can be evaluated and sampled from.

```python
import torch
import zuko

# Neural spline flow (NSF) with 3 sample features and 5 context features
flow = zuko.flows.NSF(3, 5, transforms=3, hidden_features=[128] * 3)

# Train to maximize the log-likelihood
optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3)

for x, c in trainset:
    loss = -flow(c).log_prob(x)  # -log p(x | c)
    loss = loss.mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Sample 64 points x ~ p(x | c*)
x = flow(c_star).sample((64,))
```

Alternatively, flows can be built as custom `Flow` objects.

```python
from zuko.flows import Flow, UnconditionalDistribution, UnconditionalTransform
from zuko.flows.autoregressive import MaskedAutoregressiveTransform
from zuko.distributions import DiagNormal
from zuko.transforms import RotationTransform

flow = Flow(
    transform=[
        MaskedAutoregressiveTransform(3, 5, hidden_features=(64, 64)),
        UnconditionalTransform(RotationTransform, torch.randn(3, 3)),
        MaskedAutoregressiveTransform(3, 5, hidden_features=(64, 64)),
    ],
    base=UnconditionalDistribution(
        DiagNormal,
        torch.zeros(3),
        torch.ones(3),
        buffer=True,
    ),
)
```

For more information, check out the documentation and tutorials at [zuko.readthedocs.io](https://zuko.readthedocs.io).

### Available flows

| Class   | Year | Reference |
|:-------:|:----:|-----------|
| `GMM`   | -    | [Gaussian Mixture Model](https://wikipedia.org/wiki/Mixture_model#Gaussian_mixture_model) |
| `NICE`  | 2014 | [Non-linear Independent Components Estimation](https://arxiv.org/abs/1410.8516) |
| `MAF`   | 2017 | [Masked Autoregressive Flow for Density Estimation](https://arxiv.org/abs/1705.07057) |
| `NSF`   | 2019 | [Neural Spline Flows](https://arxiv.org/abs/1906.04032) |
| `NCSF`  | 2020 | [Normalizing Flows on Tori and Spheres](https://arxiv.org/abs/2002.02428) |
| `SOSPF` | 2019 | [Sum-of-Squares Polynomial Flow](https://arxiv.org/abs/1905.02325) |
| `NAF`   | 2018 | [Neural Autoregressive Flows](https://arxiv.org/abs/1804.00779) |
| `UNAF`  | 2019 | [Unconstrained Monotonic Neural Networks](https://arxiv.org/abs/1908.05164) |
| `CNF`   | 2018 | [Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366) |
| `GF`    | 2020 | [Gaussianization Flows](https://arxiv.org/abs/2003.01941) |
| `BPF`   | 2020 | [Bernstein-Polynomial Normalizing Flows](https://arxiv.org/abs/2004.00464) |

## Contributing

If you have a question, an issue or would like to contribute, please read our [contributing guidelines](https://github.com/probabilists/zuko/blob/master/CONTRIBUTING.md).
