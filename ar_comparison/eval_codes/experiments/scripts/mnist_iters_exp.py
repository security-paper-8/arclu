import os
import sys
sys.path.append("./")
import torch
from execution import PGDParam, BaselineExecutor, Scheduler


def mnist_exp_linf_untargeted(ROOT_PATH, baselines):
    executor_list = []
    for iters in [10, 20, 40, 80, 160, 320, 640, 1000, 2000]:
        attack = "pgd"
        attack_args = PGDParam("mnist", "linf", 0.3, 0.3 /
                               iters * 1.5, iters, attack_size=1000)
        EXP_PATH = "mnist_{}_{}_{}_{}_{}".format(attack,
                                                 "targeted" if "targeted" in attack_args.keys() else "untargeted",
                                                 attack_args["eps_type"],
                                                 attack_args["eps"],
                                                 attack_args["iters"]
                                                 )
        for bl in baselines:
            print("Baseline: ", bl)
            bex = BaselineExecutor(
                bl, attack_args, os.path.join(ROOT_PATH, EXP_PATH))
            executor_list.append(bex)

    return executor_list


def mnist_exp_l2_untargeted(ROOT_PATH, baselines):
    executor_list = []
    for iters in [10, 20, 40, 80, 160, 320, 640, 1000, 2000]:
        attack = "pgd"
        attack_args = PGDParam("mnist", "l2", 2.0, 2.0 /
                               iters * 1.5, iters, attack_size=1000)
        EXP_PATH = "mnist_{}_{}_{}_{}_{}".format(attack,
                                                 "targeted" if "targeted" in attack_args.keys() else "untargeted",
                                                 attack_args["eps_type"],
                                                 attack_args["eps"],
                                                 attack_args["iters"]
                                                 )
        for bl in baselines:
            print("Baseline: ", bl)
            bex = BaselineExecutor(
                bl, attack_args, os.path.join(ROOT_PATH, EXP_PATH))
            executor_list.append(bex)

    return executor_list


def mnist_exp_l1_untargeted(ROOT_PATH, baselines):
    executor_list = []
    for iters in [10, 20, 40, 80, 160, 320, 640, 1000]:
        attack = "pgd"
        attack_args = PGDParam("mnist", "l1", 15, 4.0, iters, attack_size=1000)
        EXP_PATH = "mnist_{}_{}_{}_{}_{}".format(attack,
                                                 "targeted" if "targeted" in attack_args.keys() else "untargeted",
                                                 attack_args["eps_type"],
                                                 attack_args["eps"],
                                                 attack_args["iters"]
                                                 )
        for bl in baselines:
            print("Baseline: ", bl)
            bex = BaselineExecutor(
                bl, attack_args, os.path.join(ROOT_PATH, EXP_PATH))
            executor_list.append(bex)

    return executor_list


def mnist_exp_l0_untargeted(ROOT_PATH, baselines):
    executor_list = []
    for iters in [10, 20, 40, 80, 160, 320, 640, 1000, 2000]:
        attack = "pgd"
        attack_args = PGDParam("mnist", "l0", 25, 48, iters, attack_size=1000)
        EXP_PATH = "mnist_z{}_{}_{}_{}_{}".format(attack,
                                                  "targeted" if "targeted" in attack_args.keys() else "untargeted",
                                                  attack_args["eps_type"],
                                                  attack_args["eps"],
                                                  attack_args["iters"]
                                                  )
        for bl in baselines:
            print("Baseline: ", bl)
            bex = BaselineExecutor(
                bl, attack_args, os.path.join(ROOT_PATH, EXP_PATH))
            executor_list.append(bex)

    return executor_list


def mnist_exp():
    ROOT_PATH = "./results/mnist_iters_exp/"
    baselines = ["arclu"]

    executor_list = []
    executor_list.extend(mnist_exp_linf_untargeted(ROOT_PATH, baselines))
    executor_list.extend(mnist_exp_l2_untargeted(ROOT_PATH, baselines))
    executor_list.extend(mnist_exp_l1_untargeted(ROOT_PATH, baselines))
    executor_list.extend(mnist_exp_l0_untargeted(ROOT_PATH, baselines))

    Scheduler(list(range(torch.cuda.device_count())), executor_list)


if __name__ == "__main__":
    mnist_exp()
