import os
import sys
sys.path.append("./")
import torch
from execution import PGDParam, BaselineExecutor, Scheduler


def mnist_exp_l0_targeted(ROOT_PATH, baselines):
    executor_list = []
    for step_size in [4.0, 8.0, 16.0, 32.0, 48.0, 64.0]:
        attack = "pgd"
        attack_args = PGDParam("mnist", "l0", 25.0,
                               step_size, 1000, attack_size=1000)
        attack_args["targeted"] = ""
        EXP_PATH = "mnist_{}_{}_{}_{}_{}".format(attack,
                                                 "targeted" if "targeted" in attack_args.keys() else "untargeted",
                                                 attack_args["eps_type"],
                                                 attack_args["eps"],
                                                 attack_args["step_size"])

        for bl in baselines:
            print("Baseline: ", bl)
            bex = BaselineExecutor(
                bl, attack_args, os.path.join(ROOT_PATH, EXP_PATH))
            executor_list.append(bex)

    return executor_list


def mnist_exp_l0_untargeted(ROOT_PATH, baselines):
    executor_list = []
    for step_size in [4.0, 8.0, 16.0, 32.0, 48.0, 64.0]:
        attack = "pgd"
        attack_args = PGDParam("mnist", "l0", 25.0,
                               step_size, 1000, attack_size=1000)
        EXP_PATH = "mnist_{}_{}_{}_{}_{}".format(attack,
                                                 "targeted" if "targeted" in attack_args.keys() else "untargeted",
                                                 attack_args["eps_type"],
                                                 attack_args["eps"],
                                                 attack_args["step_size"]
                                                 )

        for bl in baselines:
            print("Baseline: ", bl)
            bex = BaselineExecutor(
                bl, attack_args, os.path.join(ROOT_PATH, EXP_PATH))
            executor_list.append(bex)

    return executor_list


def mnist_exp():
    ROOT_PATH = "./results/mnist_whitebox/"
    baselines = ["arclu"]

    executor_list = []
    executor_list.extend(mnist_exp_l0_targeted(ROOT_PATH, baselines))
    executor_list.extend(mnist_exp_l0_untargeted(ROOT_PATH, baselines))
    Scheduler(list(range(torch.cuda.device_count())), executor_list)


if __name__ == "__main__":
    mnist_exp()
