.. image:: images/banner.svg
   :class: only-light

.. image:: images/banner_dark.svg
   :class: only-dark

Zuko
====

Zuko is a Python package that implements normalizing flows in PyTorch. It relies as much as possible on distributions and transformations already provided by PyTorch. Unfortunately, the `Distribution` and `Transform` classes of :mod:`torch` are not sub-classes of :class:`torch.nn.Module`, which means you cannot send their internal tensors to GPU with :py:`.to('cuda')` or retrieve their parameters with :py:`.parameters()`.

To solve this problem, :mod:`zuko` defines two abstract classes: :class:`zuko.flows.DistributionModule` and :class:`zuko.flows.TransformModule`. The former is any `Module` whose forward pass returns a `Distribution` and the latter is any `Module` whose forward pass returns a `Transform`. Then, a normalizing flow is the composition of a list of `TransformModule` and a base `DistributionModule`. This design allows for flows that behave like distributions while retaining the benefits of `Module`. It also makes the implementations easy to understand and extend.

Installation
------------

The :mod:`zuko` package is available on `PyPI <https://pypi.org/project/zuko>`_, which means it is installable via `pip`.

.. code-block:: console

    pip install zuko

Alternatively, if you need the latest features, you can install it from the repository.

.. code-block:: console

    pip install git+https://github.com/francois-rozet/zuko

Getting started
---------------

Normalizing flows are provided in the :mod:`zuko.flows` module. To build one, supply the number of sample and context features as well as the transformations' hyperparameters. Then, feeding a context :math:`y` to the flow returns a conditional distribution :math:`p(x | y)` which can be evaluated and sampled from.

.. code-block:: python

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

References
----------

| Normalizing Flows for Probabilistic Modeling and Inference (Papamakarios et al., 2021)
| https://arxiv.org/abs/1912.02762

| Masked Autoregressive Flow for Density Estimation (Papamakarios et al., 2017)
| https://arxiv.org/abs/1705.07057

| Neural Spline Flows (Durkan et al., 2019)
| https://arxiv.org/abs/1906.04032

| Neural Autoregressive Flows (Huang et al., 2018)
| https://arxiv.org/abs/1804.00779

.. toctree::
    :caption: zuko
    :hidden:
    :maxdepth: 2

    api/index.rst

.. toctree::
    :caption: Development
    :hidden:
    :maxdepth: 1

    Contributing <https://github.com/francois-rozet/zuko/blob/master/CONTRIBUTING.md>
    Changelog <https://github.com/francois-rozet/zuko/releases>
    License <https://github.com/francois-rozet/zuko/blob/master/LICENSE>
