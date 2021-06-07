import os
import sys
sys.path.append("./")
import torch
from execution import PGDParam, BaselineExecutor, Scheduler


def light_imagenet_exp_l0_targeted(ROOT_PATH, baselines):
    executor_list = []
    for step_size in [8192, 1024 * 10, 1024 * 12, 1024 * 14, 1024 * 15]:
        attack = "pgd"
        attack_args = PGDParam("light_imagenet", "l0",
                               1000, step_size, 1000, attack_size=500)
        attack_args["targeted"] = ""
        EXP_PATH = "light_imagenet_{}_{}_{}_{}_{}".format(attack,
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


def light_imagenet_exp_l0_untargeted(ROOT_PATH, baselines):
    executor_list = []
    for step_size in [8192, 1024 * 10, 1024 * 12, 1024 * 14, 1024 * 15]:
        attack = "pgd"
        attack_args = PGDParam("light_imagenet", "l0",
                               1000, step_size, 1000, attack_size=500)
        EXP_PATH = "light_imagenet_{}_{}_{}_{}_{}".format(attack,
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


def light_imagenet_exp():
    ROOT_PATH = "./results/light_imagenet_whitebox/"
    baselines = ["arclu"]
    executor_list = []
    executor_list.extend(light_imagenet_exp_l0_targeted(ROOT_PATH, baselines))
    executor_list.extend(light_imagenet_exp_l0_untargeted(ROOT_PATH, baselines))
    Scheduler(list(range(torch.cuda.device_count())), executor_list)


if __name__ == "__main__":
    light_imagenet_exp()
