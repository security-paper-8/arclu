import os
import sys
sys.path.append("./")
import torch
from execution import AutoParam, SquareParam, FABtParam, APGDtParam, KNNParam, RandomStartParam, MultiTargetParam, FGSMParam, MIMParam, CWParam, BaselineExecutor, Scheduler


def light_imagenet_exp_auto(ROOT_PATH, baselines):
    executor_list = []
    for eps in [0.03, 0.045, 0.06, 0.25, 1.0]:
        attack = "auto"
        attack_args = AutoParam("light_imagenet", "linf", eps, attack_size=500)
        EXP_PATH = "light_imagenet_{}_{}_{}_{}".format(attack,
                                                       "targeted" if "targeted" in attack_args.keys() else "untargeted",
                                                       attack_args["eps_type"],
                                                       attack_args["eps"])

        for bl in baselines:
            print("Baseline: ", bl)
            bex = BaselineExecutor(
                bl, attack_args, os.path.join(ROOT_PATH, EXP_PATH))
            executor_list.append(bex)

    return executor_list


def light_imagenet_exp_square(ROOT_PATH, baselines):
    executor_list = []
    for eps in [0.03, 0.045, 0.06, 0.25, 1.0]:
        attack = "square"
        attack_args = SquareParam("light_imagenet", eps, attack_size=500)
        EXP_PATH = "light_imagenet_{}_{}_{}_{}".format(attack,
                                                       "targeted" if "targeted" in attack_args.keys() else "untargeted",
                                                       attack_args["eps_type"],
                                                       attack_args["eps"])

        for bl in baselines:
            print("Baseline: ", bl)
            bex = BaselineExecutor(
                bl, attack_args, os.path.join(ROOT_PATH, EXP_PATH))
            executor_list.append(bex)

    return executor_list


def light_imagenet_exp_fabt(ROOT_PATH, baselines):
    executor_list = []
    for eps in [0.03, 0.045, 0.06, 0.25, 1.0]:
        attack = "fabt"
        attack_args = FABtParam("light_imagenet", eps, attack_size=500)
        EXP_PATH = "light_imagenet_{}_{}_{}_{}".format(attack,
                                                       "targeted" if "targeted" in attack_args.keys() else "untargeted",
                                                       attack_args["eps_type"],
                                                       attack_args["eps"])

        for bl in baselines:
            print("Baseline: ", bl)
            bex = BaselineExecutor(
                bl, attack_args, os.path.join(ROOT_PATH, EXP_PATH))
            executor_list.append(bex)

    return executor_list


def light_imagenet_exp_apgdt(ROOT_PATH, baselines):
    executor_list = []
    for eps in [0.03, 0.045, 0.06, 0.25, 1.0]:
        attack = "apgdt"
        attack_args = APGDtParam("light_imagenet", eps, attack_size=500)
        EXP_PATH = "light_imagenet_{}_{}_{}_{}".format(attack,
                                                       "targeted" if "targeted" in attack_args.keys() else "untargeted",
                                                       attack_args["eps_type"],
                                                       attack_args["eps"])

        for bl in baselines:
            print("Baseline: ", bl)
            bex = BaselineExecutor(
                bl, attack_args, os.path.join(ROOT_PATH, EXP_PATH))
            executor_list.append(bex)

    return executor_list


def light_imagenet_exp_random_start(ROOT_PATH, baselines):
    executor_list = []
    for eps in [0.03, 0.045, 0.06, 0.25, 1.0]:
        attack = "random_start"
        iters = 100
        step_size = eps / iters * 1.2
        attack_args = RandomStartParam(
            "light_imagenet", "linf", eps, step_size, iters, repetition=10, attack_size=500)
        EXP_PATH = "light_imagenet_{}_{}_{}_{}".format(attack,
                                                       "targeted" if "targeted" in attack_args.keys() else "untargeted",
                                                       attack_args["eps_type"],
                                                       attack_args["eps"])

        for bl in baselines:
            print("Baseline: ", bl)
            bex = BaselineExecutor(
                bl, attack_args, os.path.join(ROOT_PATH, EXP_PATH))
            executor_list.append(bex)

    return executor_list


def light_imagenet_exp_multi_targeted(ROOT_PATH, baselines):
    executor_list = []
    for eps in [0.03, 0.045, 0.06, 0.25, 1.0]:
        attack = "multi_targeted"
        iters = 100
        step_size = eps / iters * 1.2
        attack_args = MultiTargetParam(
            "light_imagenet", "linf", eps, step_size, iters, num_classes=20, attack_size=500)
        EXP_PATH = "light_imagenet_{}_{}_{}_{}".format(attack,
                                                       "targeted" if "targeted" in attack_args.keys() else "untargeted",
                                                       attack_args["eps_type"],
                                                       attack_args["eps"])

        for bl in baselines:
            print("Baseline: ", bl)
            bex = BaselineExecutor(
                bl, attack_args, os.path.join(ROOT_PATH, EXP_PATH))
            executor_list.append(bex)

    return executor_list


def light_imagenet_exp_knn(ROOT_PATH, baselines):
    executor_list = []
    for eps in [0.03, 0.045, 0.06, 0.25, 1.0]:
        attack = "knn"
        iters = 100
        step_size = eps / iters * 1.2
        attack_args = KNNParam("light_imagenet", "linf",
                               eps, step_size, iters, attack_size=500)
        EXP_PATH = "light_imagenet_{}_{}_{}_{}".format(attack,
                                                       "targeted" if "targeted" in attack_args.keys() else "untargeted",
                                                       attack_args["eps_type"],
                                                       attack_args["eps"])

        for bl in baselines:
            print("Baseline: ", bl)
            bex = BaselineExecutor(
                bl, attack_args, os.path.join(ROOT_PATH, EXP_PATH))
            executor_list.append(bex)

    return executor_list


def light_imagenet_exp_mim(ROOT_PATH, baselines):
    executor_list = []
    for eps in [0.06]:
        attack = "mim"
        attack_args = MIMParam("light_imagenet", "linf",
                               eps, 0.001, 1000, attack_size=500)
        EXP_PATH = "light_imagenet_{}_{}_{}_{}".format(attack,
                                                       "targeted" if "targeted" in attack_args.keys() else "untargeted",
                                                       attack_args["eps_type"],
                                                       attack_args["eps"])

        for bl in baselines:
            print("Baseline: ", bl)
            bex = BaselineExecutor(
                bl, attack_args, os.path.join(ROOT_PATH, EXP_PATH))
            executor_list.append(bex)

    return executor_list


def light_imagenet_exp_cw(ROOT_PATH, baselines):
    executor_list = []
    for eps in [6.0]:
        for conf in [6400, 9600, 12800]:
            for lr in [0.01, 0.1, 0.5, 1.0]:
                attack = "cw"
                attack_args = CWParam("light_imagenet", eps, lr, 1000, initial_const=100,
                                      binary_search_steps=2, confidence=conf, attack_size=500)
                EXP_PATH = "light_imagenet_{}_{}_{}_{}_{}_{}".format(attack,
                                                                     "targeted" if "targeted" in attack_args.keys() else "untargeted",
                                                                     attack_args["eps_type"],
                                                                     attack_args["eps"], conf, lr)

                for bl in baselines:
                    print("Baseline: ", bl)
                    bex = BaselineExecutor(
                        bl, attack_args, os.path.join(ROOT_PATH, EXP_PATH))
                    executor_list.append(bex)

    return executor_list


def light_imagenet_exp():
    ROOT_PATH = "./results/light_imagenet_advanced/"
    baselines = ["arclu"]

    executor_list = []
    executor_list.extend(light_imagenet_exp_auto(ROOT_PATH, baselines))
    executor_list.extend(light_imagenet_exp_random_start(ROOT_PATH, baselines))
    executor_list.extend(
        light_imagenet_exp_multi_targeted(ROOT_PATH, baselines))
    executor_list.extend(light_imagenet_exp_knn(ROOT_PATH, ["arclu"]))
    executor_list.extend(light_imagenet_exp_cw(ROOT_PATH, baselines))
    print(list(range(torch.cuda.device_count())))
    Scheduler(list(range(torch.cuda.device_count())), executor_list)


if __name__ == "__main__":
    light_imagenet_exp()
