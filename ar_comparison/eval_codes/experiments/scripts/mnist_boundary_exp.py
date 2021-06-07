import os
import sys
sys.path.append("./")
from execution import BoundaryParam, BaselineExecutor, Scheduler
import torch


def mnist_exp_boundary_linf(ROOT_PATH, baselines):
    attack = "boundary"
    attack_args = BoundaryParam(
        "mnist", "linf", 0.3, 10, 0.8, 0.02, 1000, attack_size=1000)
    EXP_PATH = "mnist_{}_{}_{}_{}".format(attack,
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


def mnist_exp_boundary_l2(ROOT_PATH, baselines):
    attack = "boundary"
    attack_args = BoundaryParam(
        "mnist", "l2", 2.0, 10, 0.8, 0.02, 1000, attack_size=1000)
    EXP_PATH = "mnist_{}_{}_{}_{}".format(attack,
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


def mnist_exp():
    ROOT_PATH = "./results/mnist_boundary/"
    baselines = ["arclu"]

    executor_list = []
    executor_list.extend(mnist_exp_boundary_linf(ROOT_PATH, baselines))
    executor_list.extend(mnist_exp_boundary_l2(ROOT_PATH, baselines))
    Scheduler(list(range(torch.cuda.device_count())), executor_list)


if __name__ == "__main__":
    mnist_exp()
