import os
import sys
sys.path.append("./")
import torch
from execution import NESParam, BaselineExecutor, Scheduler


def light_imagenet_exp_nes_linf_untargeted(ROOT_PATH, baselines):
    attack = "nes"
    attack_args = NESParam("light_imagenet", "linf", 0.06,
                           0.01, 1000, 160, 0.02, attack_size=500)
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


def light_imagenet_exp_nes_l2_untargeted(ROOT_PATH, baselines):
    attack = "nes"
    attack_args = NESParam("light_imagenet", "l2", 6.0,
                           0.1, 1000, 160, 0.02, attack_size=500)
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
    ROOT_PATH = "./results/light_imagenet_nes/"
    baselines = ["arclu"]
    executor_list = []
    executor_list.extend(
        light_imagenet_exp_nes_linf_untargeted(ROOT_PATH, baselines))
    executor_list.extend(
        light_imagenet_exp_nes_l2_untargeted(ROOT_PATH, baselines))

    Scheduler(list(range(torch.cuda.device_count())), executor_list)


if __name__ == "__main__":
    light_imagenet_exp()
