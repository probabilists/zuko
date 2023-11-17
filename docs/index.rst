.. image:: images/banner.svg
   :class: only-light

.. image:: images/banner_dark.svg
   :class: only-dark

Zuko
====

Zuko is a Python package that implements normalizing flows in `PyTorch <https://pytorch.org>`_. It relies as much as possible on distributions and transformations already provided by PyTorch. Unfortunately, the `Distribution` and `Transform` classes of :mod:`torch` are not sub-classes of :class:`torch.nn.Module`, which means you cannot send their internal tensors to GPU with :py:`.to('cuda')` or retrieve their parameters with :py:`.parameters()`. Worse, the concepts of conditional distribution and transformation, which are essential for probabilistic inference, are impossible to express.


To solve these problems, :mod:`zuko` defines two concepts: the :class:`zuko.flows.core.LazyDistribution` and :class:`zuko.flows.core.LazyTransform`, which are any modules whose forward pass returns a `Distribution` or `Transform`, respectively. Because the creation of the actual distribution/transformation is delayed, an eventual condition can be easily taken into account. This design enables lazy distributions, including normalizing flows, to act like distributions while retaining features inherent to modules, such as trainable parameters. It also makes the implementations easy to understand and extend.

Installation
------------

The :mod:`zuko` package is available on `PyPI <https://pypi.org/project/zuko>`_, which means it is installable via `pip`.

.. code-block:: console

    pip install zuko

Alternatively, if you need the latest features, you can install it from the repository.

.. code-block:: console

    pip install git+https://github.com/probabilists/zuko

Getting started
---------------

Normalizing flows are provided in the :mod:`zuko.flows` module. To build one, supply the number of sample and context features as well as the transformations' hyperparameters. Then, feeding a context :math:`c` to the flow returns a conditional distribution :math:`p(x | c)` which can be evaluated and sampled from.

.. code-block:: python

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

Alternatively, flows can be built as custom :class:`zuko.flows.core.Flow` objects.

.. code-block:: python

    from zuko.flows import Flow, MaskedAutoregressiveTransform, Unconditional
    from zuko.distributions import DiagNormal
    from zuko.transforms import RotationTransform

    flow = Flow(
        transform=[
            MaskedAutoregressiveTransform(3, 5, hidden_features=(64, 64)),
            Unconditional(RotationTransform, torch.randn(3, 3)),
            MaskedAutoregressiveTransform(3, 5, hidden_features=(64, 64)),
        ],
        base=Unconditional(
            DiagNormal,
            torch.zeros(3),
            torch.ones(3),
            buffer=True,
        ),
    )

For more information, check out the :doc:`tutorials <../tutorials>` or the :doc:`API <../api>`.

References
----------

| NICE: Non-linear Independent Components Estimation (Dinh et al., 2014)
| https://arxiv.org/abs/1410.8516

| Variational Inference with Normalizing Flows (Rezende et al., 2015)
| https://arxiv.org/abs/1505.05770

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

    tutorials.rst
    api.rst

.. toctree::
    :caption: Development
    :hidden:
    :maxdepth: 1

    Contributing <https://github.com/probabilists/zuko/blob/master/CONTRIBUTING.md>
    Changelog <https://github.com/probabilists/zuko/releases>
    License <https://github.com/probabilists/zuko/blob/master/LICENSE>
