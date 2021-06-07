import os
import sys
sys.path.append("./")
import torch
from execution import PGDDisperseParam, MultiTargetDisperseParam, BaselineExecutor, Scheduler


def mnist_exp_linf_untargeted(ROOT_PATH, baselines):

    executor_list = []
    for ztm in [0.1, 0.2, 0.4, 0.8, 1.0]:
        for ac in [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0]:
            attack = "pgddisperse"
            attack_args = PGDDisperseParam("mnist", "linf", 0.4, 0.0001, 5000,
                                           zero_threshold_multiplier=ztm, adv_coeff=ac, attack_size=1000)
            EXP_PATH = "mnist_{}_{}_{}_{}_{}_{}".format(attack,
                                                        "targeted" if "targeted" in attack_args.keys() else "untargeted",
                                                        attack_args["eps_type"],
                                                        attack_args["eps"],
                                                        attack_args["zero_threshold_multiplier"],
                                                        attack_args["adv_coeff"]
                                                        )

            for bl in baselines:
                print("Baseline: ", bl)
                bex = BaselineExecutor(
                    bl, attack_args, os.path.join(ROOT_PATH, EXP_PATH))
                executor_list.append(bex)

    return executor_list


def mnist_exp_multi_target_linf_untargeted(ROOT_PATH, baselines):
    executor_list = []
    for ztm in [0.1, 0.2, 0.4, 0.8, 1.0]:
        for ac in [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0]:
            attack = "multidisperse"
            attack_args = MultiTargetDisperseParam("mnist", "linf", 0.4, 0.0001, 5000,
                                                   num_classes=10, zero_threshold_multiplier=ztm,
                                                   adv_coeff=ac, attack_size=1000)
            EXP_PATH = "mnist_{}_{}_{}_{}_{}_{}".format(attack,
                                                        "targeted" if "targeted" in attack_args.keys() else "untargeted",
                                                        attack_args["eps_type"],
                                                        attack_args["eps"],
                                                        attack_args["zero_threshold_multiplier"],
                                                        attack_args["adv_coeff"]
                                                        )

            for bl in baselines:
                print("Baseline: ", bl)
                bex = BaselineExecutor(
                    bl, attack_args, os.path.join(ROOT_PATH, EXP_PATH))
                executor_list.append(bex)

    return executor_list


def mnist_exp():
    ROOT_PATH = "./results/mnist_whitebox_disperse/"
    baselines = ["arclu"]
    executor_list = []

    executor_list.extend(mnist_exp_linf_untargeted(ROOT_PATH, baselines))
    executor_list.extend(
        mnist_exp_multi_target_linf_untargeted(ROOT_PATH, baselines))
    Scheduler(list(range(torch.cuda.device_count())), executor_list)


if __name__ == "__main__":
    mnist_exp()
