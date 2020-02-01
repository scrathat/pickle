#!/usr/bin/env python 3
"""
Create pickle files from raw datasets
"""

__version__ = "0.0.1"

import _pickle as cPickle
import argparse
import logging
import os
import time
from argparse import ArgumentError

import numpy as np
from tqdm import tqdm

from speech_dataset import SpokenLanguage

DATASETS = [
    "ALL",
    "spokendigit",
    "isolet",
    "spokenlanguage",
    "librispeech",
    "millionsong",
    "musicgenres",
    "urbansound",
    "sicksounds",
    "audioset",
    "commonvoice",
    "spokenwiki",
]


def save(dataset_name, dataset_type, data, labels):
    logging.info("Saving progress...")
    os.makedirs(dataset_name, exist_ok=True)
    filepath = f"{dataset_name}/{dataset_name}.{dataset_type}"
    cPickle.dump(data, open(f"{filepath}.pkl", "wb"))
    np.savetxt(f"{filepath}.solution", np.asarray(labels), fmt="%d")


def main(args):
    start_time = time.time()

    for s in ["train", "test"]:
        path = os.path.join(args.path, f"{s}/{s}")

        logging.info(f"Reading {s} dataset from path: {path}")
        dataset = SpokenLanguage(path)

        logging.info(f"Read {s} dataset of length: {len(dataset)}")

        data = []
        labels = []
        for i, sample in enumerate(tqdm(dataset)):
            (x, y) = sample
            data.append(x)
            labels.append(y)
            if i > 0 and i % 10000 == 0:
                save(args.dataset, s, data, labels)

        save(args.dataset, s, data, labels)

    end_time = time.time()
    logging.info(f"Running time: {end_time - start_time}")


if __name__ == "__main__":
    description = f"(Download) and pickle datasets: {DATASETS}"
    parser = argparse.ArgumentParser(description=description)
    dataset_arg = parser.add_argument("dataset", help="Name of the dataset to download")
    path_arg = parser.add_argument(
        "path",
        help="Path to dir containing downloaded datasets. Results are saved here",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Increase verbosity"
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s (version {__version__})"
    )
    args = parser.parse_args()

    logging_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=logging_level,
        format="%(asctime)s %(levelname)s %(filename)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not os.path.isdir(args.path):
        raise ArgumentError(path_arg, f"{args.path} is not a valid path")
    if not os.access(args.path, os.R_OK):
        raise ArgumentError(path_arg, f"{args.path} is not a readable dir")
    if args.dataset not in DATASETS:
        raise ArgumentError(dataset_arg, f"{args.dataset} is not in list of datasets")

    main(args)
