# Backfire - Normalizing flows in PyTorch

`backfire` is a Python package that implements normalizing flows in PyTorch. The philosophy of `backfire` is to rely as much as possible on distributions and transformations already existing in PyTorch. Unfortunately, the `Distribution` and `Transformation` classes of `torch` are not sub-classes of `nn.Module`, which means they don't implement a forward method, you cannot send their internal tensors to GPU with `.to('cuda')` or even get their parameters with `.parameters()`.

To solve this problem, `backfire` defines two abstract classes: `DistributionModule` and `TransformModule`. The former is any `nn.Module` whose forward pass returns a `Distribution`. Similarly, the latter is any `nn.Module` whose forward pass returns a `Transform`. Then, a normalizing flow is simply a `nn.Module` that is constructed from a list of `TransformModule` and a base `DistributionModule` and its parameters are handled just like any neural network. This design allows for flow implementations that are easy to understand and extend.

## Installation

The `backfire` package is available on [PyPI](https://pypi.org/project/backfire), which means it is installable via `pip`.

```
pip install backfire
```

Alternatively, if you need the latest features, you can install it from the repository.

```
pip install git+https://github.com/francois-rozet/backfire
```

## Contributing

If you have a question, an issue or would like to contribute, please read our [contributing guidelines](CONTRIBUTING.md).

## Documentation

The documentation is made with [Sphinx](https://www.sphinx-doc.org) and [Furo](https://github.com/pradyunsg/furo) and is hosted at [francois-rozet.github.io/backfire](https://francois-rozet.github.io/backfire).
