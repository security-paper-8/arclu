import os
import sys
sys.path.append("./")
import torch
from execution import NESParam, BaselineExecutor, Scheduler


def mnist_exp_nes_linf_untargeted(ROOT_PATH, baselines):
    executor_list = []
    for eps in [0.3, 0.4]:
        attack = "nes"
        attack_args = NESParam("mnist", "linf", eps, 0.01,
                               1000, 160, 0.02, attack_size=1000)
        EXP_PATH = "mnist_{}_{}_{}_{}".format(attack,
                                              "targeted" if "targeted" in attack_args.keys() else "untargeted",
                                              attack_args["eps_type"],
                                              attack_args["eps"])

        for bl in baselines:
            print("Baseline: ", bl)
            bex = BaselineExecutor(
                bl, attack_args, os.path.join(ROOT_PATH, EXP_PATH))
            executor_list.append(bex)

    return executor_list


def mnist_exp_nes_l2_untargeted(ROOT_PATH, baselines):
    attack = "nes"
    attack_args = NESParam("mnist", "l2", 2.0, 0.1, 500,
                           160, 0.02, attack_size=1000)
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
    ROOT_PATH = "./results/mnist_nes/"
    baselines = ["arclu"]

    executor_list = []
    executor_list.extend(mnist_exp_nes_linf_untargeted(ROOT_PATH, baselines))
    executor_list.extend(mnist_exp_nes_l2_untargeted(ROOT_PATH, baselines))

    Scheduler(list(range(torch.cuda.device_count())), executor_list)


if __name__ == "__main__":
    mnist_exp()
