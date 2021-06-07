import os
import sys
sys.path.append("./")
import torch
from execution import PGDParam, BaselineExecutor, Scheduler


def mnist_exp_linf_untargeted(ROOT_PATH, baselines):
    attack = "pgd"
    attack_args = PGDParam("mnist", "linf", 0.3, 0.0001, 4000, attack_size=50)
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
    ROOT_PATH = "./results/mnist_quick_test/"
    baselines = ["arclu"]
    executor_list = []
    executor_list.extend(mnist_exp_linf_untargeted(ROOT_PATH, baselines))
    Scheduler(list(range(torch.cuda.device_count())), executor_list)


if __name__ == "__main__":
    mnist_exp()
