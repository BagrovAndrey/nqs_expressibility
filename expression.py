#!/usr/bin/env python3

from copy import deepcopy
import datetime

# import cProfile
from shutil import copyfile
import scipy
import json
import math
import os
import os.path
import pickle
import re
import sys
import tempfile
import time
from typing import Dict, List, Tuple, Optional
import random_state_generator as rsg

import numpy as np
import torch
import torch.utils.data

import scipy.io as sio
from scipy.special import comb
from itertools import combinations

# "Borrowed" from pytorch/torch/serialization.py.
# All credit goes to PyTorch developers.
def _with_file_like(f, mode, body):
    """
    Executes a body function with a file object for f, opening
    it in 'mode' if it is a string filename.
    """
    new_fd = False
    if (
        isinstance(f, str)
        or (sys.version_info[0] == 2 and isinstance(f, unicode))
        or (sys.version_info[0] == 3 and isinstance(f, pathlib.Path))
    ):
        new_fd = True
        f = open(f, mode)
    try:
        return body(f)
    finally:
        if new_fd:
            f.close()


def _make_checkpoints_for(n: int, steps: int = 10):
    if n <= steps:
        return list(range(0, n))
    important_iterations = list(range(0, n, n // steps))
    if important_iterations[-1] != n - 1:
        important_iterations.append(n - 1)
    return important_iterations


def train(ψ, train_set, gpu, lr, **config):
    if gpu:
        ψ = ψ.cuda()
    
    epochs = config["epochs"]
    optimiser = config['optimiser'](ψ)

    loss_fn = config["loss"]
    check_frequency = config["frequency"]
    load_best = True
    verbose = config["verbose"]
    print_info = print if verbose else lambda *_1, **_2: None

    print_info("Training on {} spin configurations...".format(train_set[0].size(0)))
    start = time.time()

    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*train_set),
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=1,
    )
    checkpoints = set(_make_checkpoints_for(epochs, steps=100)) if verbose else set()
    train_loss_history = []

    def training_loop():
        update_count = 0
        for epoch_index in range(epochs):
            important = epoch_index in checkpoints
            if important:
                losses = []
                accuracies = []
            for batch_index, (batch_x, batch_y, batch_weight) in enumerate(dataloader):
                if gpu:
                    batch_x, batch_y, batch_weight = batch_x.cuda(), batch_y.cuda(), batch_weight.cuda()
                optimiser.zero_grad()
                predicted = ψ(batch_x)
                loss = loss_fn(predicted, batch_y, batch_weight)
                loss.backward()
                optimiser.step()
                update_count += 1
                train_loss_history.append(
                    (update_count, epoch_index, loss.item())
                )
                if important:
                    losses.append(loss.item())

            if important:
                losses = torch.tensor(losses)
                accuracies = torch.tensor(accuracies)
                print_info(
                    "{:3d}%: train loss     = {:.3e} ± {:.2e}; train loss     ∈ [{:.3e}, {:.3e}]".format(
                        100 * (epoch_index + 1) // epochs,
                        torch.mean(losses).item(),
                        torch.std(losses).item(),
                        torch.min(losses).item(),
                        torch.max(losses).item(),
                    )
                )
        return False

    stopped_early = training_loop()
    finish = time.time()
    print_info("Finished training in {:.2f} seconds!".format(finish - start))
    if gpu:
        ψ = ψ.cpu()
    return ψ, train_loss_history


def import_network(filename: str):
    import importlib

    module_name, extension = os.path.splitext(os.path.basename(filename))
    module_dir = os.path.dirname(filename)
    if extension != ".py":
        raise ValueError(
            "Could not import the network from {!r}: not a Python source file.".format(
                filename
            )
        )
    if not os.path.exists(filename):
        raise ValueError(
            "Could not import the network from {!r}: no such file or directory".format(
                filename
            )
        )
    sys.path.insert(0, module_dir)
    module = importlib.import_module(module_name)
    sys.path.pop(0)
    return module.Net
 
def overlap(ψ, samples, target, gpu):
    if gpu:
        ψ = ψ.cuda()
        samples = samples.cuda()
        target = target.cuda()
    overlap = 0.0
    norm_bra = 0.0
    norm_ket = 0.0
    size = samples.size()[0]
    for idxs in np.split(np.arange(size), np.arange(0, size, 10000))[1:]:
        predicted = ψ(samples[idxs]).cpu().type(torch.FloatTensor)[:, 0]
        overlap += torch.sum(predicted.type(torch.FloatTensor) * target.type(torch.FloatTensor)).item()
        norm_bra += torch.sum(torch.sqr(predicted.type(torch.FloatTensor))).item()
        norm_ket += torch.sum(torch.sqr(target.type(torch.FloatTensor))).item()
    if gpu:
        ψ = ψ.cpu()
        samples = samples.cpu()
        target = target.cpu()

    return overlap / np.sqrt(norm_bra) / np.sqrt(norm_ket)

def generate_dateset(number_spins, magnetisation):
    hamming = (number_spins - magnetisation) // 2
    dimension = int(scipy.special.binom(number_spins, hamming))

    vectors = np.array(rsg.generate_binaries(number_spins, hamming))
    vectors = np.array([rsg.spin2array(vec, number_spins) for vec in vectors])
    amplitudes = np.array(rsg.generate_amplitude(dimension))

    dataset = tuple(
        torch.from_numpy(x) for x in [vectors, amplitudes]
    )
    return dataset

def try_one_dataset(n_spins, magnetisation, output, Net, number_runs, train_options, 
                    lr = 0.0003, gpu = False):
    dataset = generate_dateset(n_spins, magnetisation)

    print(dataset)
    class Loss(object):
        def __init__(self):
            self._fn = torch.nn.MSELoss()

        def __call__(self, predicted, expected):
            return self._fn(predicted, expected)

    loss_fn = Loss()
    train_options = deepcopy(train_options)
    train_options["loss"] = loss_fn
    train_options["optimiser"] = eval(train_options["optimiser"][:-1] + str(', lr = ') + str(lr) + ')')
    print(train_options["optimiser"])
    stats = []
    for i in range(number_runs):
        module = Net(dataset[0].size(1))

        module, train_history = train(
            module, dataset, gpu, lr, **train_options
        )

        if gpu:
            module = module.cuda()
            dataset = (dataset[0].cuda(), dataset[1])

        predicted = torch.zeros([0, 1], dtype=torch.float32)

        with torch.no_grad():
            size = dataset[0].size()[0]
            for idxs in np.split(np.arange(size), np.arange(0, size, 10000))[1:]:
                predicted_local = module(dataset[0][idxs]).cpu()
                predicted = torch.cat((predicted, predicted_local), dim = 0)

            total_loss = 0.0
            for idxs in np.split(np.arange(size), np.arange(0, size, 10000))[1:]:
                total_loss += loss_fn(predicted[idxs], dataset[1][idxs]).item()

        best_overlap = overlap(module, *dataset, gpu)
        print('total dataset overlap = ' + str(best_overlap) + 'total dataset accuracy = ' + str(rest_accuracy))

        if gpu:
            module = module.cpu()
            dataset = (dataset[0].cpu(), dataset[1])
        stats.append((best_overlap, total_loss))
        print("train_loss = {:.10e}, train_overlap = {:.10e}".format(total_loss, best_overlap))
    
    stats = np.array(stats)
    return np.vstack((np.mean(stats, axis=0), np.std(stats, axis=0))).T.reshape(-1)


def main():
    if not len(sys.argv) == 2:
        print(
            "Usage: python3 {} <path-to-json-config>".format(sys.argv[0]),
            file=sys.stderr,
        )
        sys.exit(1)
    config = _with_file_like(sys.argv[1], "r", json.load)
    output = config["output"]
    number_spins = config["number_spins"]
    magnetisation = config["magnetisation"]
    number_runs = config["number_runs"]
    gpu = config["gpu"]
    lr = config["lr"]
    mag = config["magnetisation"]
    Net = import_network(config["model"])
    if config["use_jit"]:
        _dummy_copy_of_Net = Net
        Net = lambda n: torch.jit.trace(
            _dummy_copy_of_Net(n), torch.rand(config["training"]["batch_size"], n)
        )

    os.makedirs(output, exist_ok=True)
    results_filename = os.path.join(output, "results.dat")
    copyfile(sys.argv[1], os.path.join(output, "config.dat"))  # copy config file to the same location where the results.txt file is
    if os.path.isfile(results_filename):
        results_file = open(results_filename, "a")
    else:
        results_file = open(results_filename, "w")

    results_file.write("# process with pid = " + str(os.getpid()) + ', launched at ' + str(datetime.datetime.now()) + '\n')
    results_file.write("# <overlap> <doverlap> <loss> <dloss> \n")

    local_output = output
    os.makedirs(local_output, exist_ok=True)
    local_result = try_one_dataset(
        number_spins, mag, local_output, Net, number_runs, config["training"], lr = lr, gpu = gpu,
    )
    with open(results_filename, "a") as results_file:
        results_file.write(
                ("{:.3f} {:.5f}" + " {:.10e}" * 22 + "\n").format(j2, rt, *tuple(local_result))
        )
    return


if __name__ == "__main__":
    main()