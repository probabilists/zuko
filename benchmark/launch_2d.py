#!usr/bin/env python

import argparse
import inspect
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import wandb

from dawgz import job, schedule
from functools import partial
from PIL import Image
from tqdm import tqdm

import zuko


def two_moons(n: int, sigma: float = 1e-1) -> np.ndarray:
    theta = 2 * np.pi * np.random.rand(n)
    shift = (theta > np.pi) - 1 / 2

    x = np.stack(
        (
            np.cos(theta) + shift,
            np.sin(theta) + shift / 2,
        ),
        axis=-1,
    )

    return x + sigma * np.random.randn(n, 2)


def two_circles(n: int, sigma: float = 1e-1) -> np.ndarray:
    theta = 2 * np.pi * np.random.rand(n)
    r = np.where(np.random.rand(n) < 0.5, 0.75, 1.5)

    x = np.stack(
        (
            r * np.cos(theta),
            r * np.sin(theta),
        ),
        axis=-1,
    )

    return x + sigma * np.random.randn(n, 2)


def two_spirals(n: int, sigma: float = 5e-2) -> np.ndarray:
    theta = 3 * np.pi * np.sqrt(np.random.rand(n))
    r = theta / 2 / np.pi * np.sign(np.random.randn(n))

    x = np.stack(
        (
            r * np.cos(theta),
            r * np.sin(theta),
        ),
        axis=-1,
    )

    return x + sigma * np.random.randn(n, 2)


def eight_gaussians(n: int, sigma: float = 1e-1) -> np.ndarray:
    theta = 2 * np.pi * np.random.randint(0, 8, n) / 8

    x = np.stack(
        (
            np.cos(theta),
            np.sin(theta),
        ),
        axis=-1,
    )

    return x + sigma * np.random.randn(n, 2)


def pinwheel(n: int) -> np.ndarray:
    theta = 2 * np.pi * np.random.randint(0, 5, n) / 5

    a = 0.25 * np.random.randn(n) + 1
    b = 0.1 * np.random.randn(n)

    theta = theta + np.exp(a - 1)

    x = np.stack(
        (
            a * np.cos(theta) - b * np.sin(theta),
            a * np.sin(theta) + b * np.cos(theta),
        ),
        axis=-1,
    )

    return x


def checkerboard(n: int) -> np.ndarray:
    u = np.random.randint(0, 4, n)
    v = np.random.randint(0, 2, n) * 2 + u % 2

    x = np.stack((u - 2, v - 2), axis=-1)

    return x + np.random.rand(n, 2)


FLOWS = {
    "GMM": partial(zuko.flows.GMM, components=16),
    "MAF": zuko.flows.MAF,
    "NAF": zuko.flows.NAF,
    "UNAF": zuko.flows.UNAF,
    "SOSPF": zuko.flows.SOSPF,
    "NSF": zuko.flows.NSF,
    "FFJORD": zuko.flows.CNF,
    "GF": zuko.flows.GF,
}

DATASETS = {
    "moons": two_moons,
    "circles": two_circles,
    "spirals": two_spirals,
    "gaussians": eight_gaussians,
    "pinwheel": pinwheel,
    "checkerboard": checkerboard,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--transforms", type=int, default=3)
    parser.add_argument("--hidden-features", type=int, nargs="+", default=(64, 64))
    parser.add_argument("--residual", default=False, action="store_true")
    parser.add_argument("--samples", type=int, default=16384)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    jobs = []

    for arch, build in FLOWS.items():
        for dataset, sample in DATASETS.items():

            @job(name=f"{arch}: {dataset}", cpus=4, ram="16GB", time="06:00:00")
            def experiment(arch=arch, build=build, dataset=dataset, sample=sample):
                np.random.seed(args.seed)
                torch.manual_seed(args.seed)

                # Data
                trainset = torch.from_numpy(sample(args.samples)).float()
                testset = torch.from_numpy(sample(args.samples)).float()

                # Flow
                if "transforms" in inspect.signature(build).parameters:
                    kwargs = dict(transforms=args.transforms, hidden_features=args.hidden_features)
                else:
                    kwargs = dict(hidden_features=args.hidden_features)

                try:
                    flow = build(trainset.shape[-1], residual=args.residual, **kwargs)
                except TypeError:
                    flow = build(trainset.shape[-1], **kwargs)

                optimizer = torch.optim.AdamW(
                    flow.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
                )

                run = wandb.init(
                    project="zuko-benchmark-2d",
                    config=dict(arch=arch, dataset=dataset, **vars(args)),
                )

                # Training
                for epoch in tqdm(range(args.epochs + 1), ncols=88):
                    order = torch.randperm(len(trainset))

                    train_losses = []
                    test_losses = []

                    flow.train()

                    for x in trainset[order].split(args.batch_size):
                        loss = -flow().log_prob(x).mean()
                        loss.backward()

                        optimizer.step()
                        optimizer.zero_grad()

                        train_losses.append(loss.detach())

                    flow.eval()

                    with torch.no_grad():
                        for x in testset.split(args.batch_size):
                            loss = -flow().log_prob(x).mean()

                            test_losses.append(loss.detach())

                    train_loss = torch.stack(train_losses).mean()
                    test_loss = torch.stack(test_losses).mean()

                    if epoch % 8 == 0:
                        start = time.perf_counter()
                        x = flow().sample((args.samples,))
                        end = time.perf_counter()

                        h, _, _ = np.histogram2d(
                            *x.T, bins=64, range=((-2, 2), (-2, 2)), density=True
                        )

                        img = plt.cm.ScalarMappable().to_rgba(h, bytes=True)
                        img = Image.fromarray(img)

                        run.log({
                            "train_loss": train_loss,
                            "test_loss": test_loss,
                            "speed": args.samples / (end - start),
                            "histogram": wandb.Image(img),
                        })
                    else:
                        run.log({
                            "train_loss": train_loss,
                            "test_loss": test_loss,
                        })

                run.finish()

            jobs.append(experiment)

    schedule(
        *jobs,
        name="Benchmark 2D",
        backend="slurm",
        export="ALL",
        env=["export WANDB_SILENT=true"],
    )
