import os
import sys
sys.path.append("./")
from execution import BoundaryParam, BaselineExecutor, Scheduler
import torch


def light_imagenet_exp_boundary_linf(ROOT_PATH, baselines):
    attack = "boundary"
    attack_args = BoundaryParam(
        "light_imagenet", "linf", 0.06, 10, 0.8, 0.02, 150, attack_size=500)
    EXP_PATH = "light_imagenet_{}_{}_{}_{}".format(attack,
                                                   "targeted" if "targeted" in attack_args.keys() else "untargeted",
                                                   attack_args["eps_type"],
                                                   attack_args["eps"])


    executor_list = []
    for bl in baselines:
        print("Baseline: ", bl)
        bex = BaselineExecutor(
            bl, attack_args, os.path.join(ROOT_PATH, EXP_PATH))
        executor_list.append(bex)

    return executor_list


def light_imagenet_exp_boundary_l2(ROOT_PATH, baselines):
    attack = "boundary"
    attack_args = BoundaryParam(
        "light_imagenet", "l2", 6.0, 10, 3.0, 0.02, 150, attack_size=500)
    EXP_PATH = "light_imagenet_{}_{}_{}_{}".format(attack,
                                                   "targeted" if "targeted" in attack_args.keys() else "untargeted",
                                                   attack_args["eps_type"],
                                                   attack_args["eps"])


    executor_list = []
    for bl in baselines:
        print("Baseline: ", bl)
        bex = BaselineExecutor(
            bl, attack_args, os.path.join(ROOT_PATH, EXP_PATH))
        executor_list.append(bex)

    return executor_list


def light_imagenet_exp():
    ROOT_PATH = "./results/light_imagenet_boundary/"
    baselines = ["arclu"]

    executor_list = []
    executor_list.extend(light_imagenet_exp_boundary_linf(ROOT_PATH, baselines))
    executor_list.extend(light_imagenet_exp_boundary_l2(ROOT_PATH, baselines))
    Scheduler(list(range(torch.cuda.device_count())), executor_list)


if __name__ == "__main__":
    light_imagenet_exp()
