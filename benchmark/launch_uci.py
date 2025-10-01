#!usr/bin/env python

import argparse
import inspect
import numpy as np
import torch
import wandb

from dawgz import context, job, schedule
from functools import partial
from tqdm import tqdm

import zuko


def power():
    return np.load("POWER.npy")


def gas():
    return np.load("GAS.npy")


def hepmass():
    return np.load("HEPMASS.npy")


def bsds300():
    return np.load("BSDS300.npy") * 10


FLOWS = {
    "GMM": partial(zuko.flows.GMM, components=16),
    "MAF (AR)": zuko.flows.MAF,
    "MAF (C)": partial(zuko.flows.MAF, passes=2),
    "NAF (AR)": zuko.flows.NAF,
    "NAF (C)": partial(zuko.flows.NAF, passes=2),
    "UNAF (AR)": zuko.flows.UNAF,
    "UNAF (C)": partial(zuko.flows.UNAF, passes=2),
    "SOSPF (AR)": zuko.flows.SOSPF,
    "SOSPF (C)": partial(zuko.flows.SOSPF, passes=2),
    "NSF (AR)": zuko.flows.NSF,
    "NSF (C)": partial(zuko.flows.NSF, passes=2),
    "FFJORD": partial(zuko.flows.CNF, exact=False),
    "GF": zuko.flows.GF,
}

DATASETS = {
    "POWER": power,
    "GAS": gas,
    "HEPMASS": hepmass,
    "BSDS300": bsds300,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--transforms", type=int, default=3)
    parser.add_argument("--randperm", default=False, action="store_true")
    parser.add_argument("--hidden-features", type=int, nargs="+", default=(512, 512))
    parser.add_argument("--residual", default=False, action="store_true")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=256)
    parser.add_argument("--epoch-size", type=int, default=65536)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    jobs = []

    for arch, build in FLOWS.items():
        for dataset, load in DATASETS.items():
            @job(name=f"{arch}: {dataset}", cpus=4, gpus=1, ram="16GB", time="24:00:00")
            def experiment(arch=arch, build=build, dataset=dataset, load=load):
                np.random.seed(args.seed)
                torch.manual_seed(args.seed)

                # Data
                data = load()
                split = int(0.9 * len(data))

                trainset = torch.from_numpy(data[:split]).float()
                testset = torch.from_numpy(data[split:]).float()

                # Flow
                if "transforms" in inspect.signature(build).parameters:
                    kwargs = dict(transforms=args.transforms, hidden_features=args.hidden_features)
                else:
                    kwargs = dict(hidden_features=args.hidden_features)

                try:
                    flow = build(
                        trainset.shape[-1],
                        randperm=args.randperm,
                        residual=args.residual,
                        **kwargs,
                    )
                except TypeError:
                    flow = build(trainset.shape[-1], **kwargs)
                finally:
                    flow.cuda()

                optimizer = torch.optim.AdamW(
                    flow.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
                )
                scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 1.0, 0.0, args.epochs)

                run = wandb.init(
                    project="zuko-benchmark-uci",
                    config=dict(arch=arch, dataset=dataset, **vars(args)),
                )

                # Training
                for epoch in tqdm(range(args.epochs + 1), ncols=88):
                    train_losses = []
                    test_losses = []

                    flow.train()

                    order = torch.randperm(len(trainset))[: args.epoch_size]

                    for x in trainset[order].split(args.batch_size):
                        x = x.cuda(non_blocking=True)

                        loss = -flow().log_prob(x).mean()
                        loss.backward()

                        optimizer.step()
                        optimizer.zero_grad()

                        train_losses.append(loss.detach())

                    flow.eval()

                    with torch.no_grad():
                        for x in testset.split(args.batch_size):
                            x = x.cuda(non_blocking=True)

                            loss = -flow().log_prob(x).mean()

                            test_losses.append(loss.detach())

                    train_loss = torch.stack(train_losses).mean()
                    test_loss = torch.stack(test_losses).mean()

                    if epoch % 8 == 0:
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)

                        start.record()
                        x = flow().sample((4096,))
                        end.record()

                        torch.cuda.synchronize()

                        run.log({
                            "train_loss": train_loss,
                            "test_loss": test_loss,
                            "speed": 4096 / start.elapsed_time(end),
                        })
                    else:
                        run.log({
                            "train_loss": train_loss,
                            "test_loss": test_loss,
                        })

                    scheduler.step()

                run.finish()

            jobs.append(experiment)

    schedule(
        *jobs,
        name="Benchmark UCI",
        backend="slurm",
        export="ALL",
        env=["export WANDB_SILENT=true"],
    )
