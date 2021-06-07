import os
import argparse
from datetime import datetime

import torch
import torch.utils.data


import numpy as np

from models import MNISTModel
import attacks
from trainers.trainer import MNISTTrainer
from trainers.light_imagenet_trainer import LightImageNetTrainer
from common import compute_thresholds_each_class, get_y_mu_encoding, split_x_adv, compute_class_mean, test_model, load_data


def make_argpase():
    parse = argparse.ArgumentParser()
    parse.add_argument("--dataset", type=str,
                       choices=["mnist", "light_imagenet"])
    parse.add_argument("--FPR", required=True, type=float)
    parse.add_argument("--epochs", default=100, type=int)
    parse.add_argument("--lr", type=float, default=0.001)
    parse.add_argument("--THRESHOLD_UPDATE_PERIOD_IN_STEP",
                       type=int, default=50)
    parse.add_argument("--NUM_THRESHOLDS_UPDATE_BATCH", type=int, default=5)
    parse.add_argument("--batch_size", type=int, default=100)

    return parse


def main():
    parse = make_argpase()
    args = parse.parse_args()
    if args.dataset == "mnist":
        trainer = MNISTTrainer(args)
        trainer.train()
    elif args.dataset == "light_imagenet":
        trainer = LightImageNetTrainer(args)
        trainer.train()


if __name__ == "__main__":
    main()
