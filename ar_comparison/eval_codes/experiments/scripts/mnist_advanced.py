import os
import sys
sys.path.append("./")
import torch
from execution import AutoParam, SquareParam, FABtParam, APGDtParam, KNNParam, RandomStartParam, MultiTargetParam, CWParam, BaselineExecutor, Scheduler, PGDParam


def mnist_exp_auto(ROOT_PATH, baselines):
    executor_list = []
    for eps in [0.3, 0.35, 0.375, 0.4, 0.6, 1.0]:
        attack = "auto"
        attack_args = AutoParam("mnist", "linf", eps, attack_size=1000)
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


def mnist_exp_square(ROOT_PATH, baselines):
    executor_list = []
    for eps in [0.35, 0.375, 0.4, 0.6, 1.0]:
        attack = "square"
        attack_args = SquareParam("mnist", eps, attack_size=1000)
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


def mnist_exp_fabt(ROOT_PATH, baselines):
    executor_list = []
    for eps in [0.35, 0.375, 0.4, 0.6, 1.0]:
        attack = "fabt"
        attack_args = FABtParam("mnist", eps, attack_size=1000)
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


def mnist_exp_apgdt(ROOT_PATH, baselines):
    executor_list = []
    for eps in [0.35, 0.375, 0.4, 0.6, 1.0]:
        attack = "apgdt"
        attack_args = APGDtParam("mnist", eps, attack_size=1000)
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


def mnist_exp_random_start(ROOT_PATH, baselines):
    executor_list = []
    for eps in [0.4]:
        attack = "random_start"
        attack_args = RandomStartParam(
            "mnist", "linf", eps, 0.0001, 5000, repetition=10, attack_size=1000)
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


def mnist_exp_multi_targeted(ROOT_PATH, baselines):
    executor_list = []
    for eps in [0.3, 0.35, 0.375, 0.4, 0.6, 1.0]:
        attack = "multi_targeted"
        attack_args = MultiTargetParam("mnist", "linf", eps, 0.0001, int(
            eps / 0.0001 * 1.5), num_classes=10, attack_size=1000)
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


def mnist_exp_knn(ROOT_PATH, baselines):
    executor_list = []
    for eps in [0.35, 0.375, 0.4, 0.6, 1.0]:
        attack = "knn"
        attack_args = KNNParam("mnist", "linf", eps, 0.0001, int(
            eps / 0.0001 * 1.5), attack_size=1000)
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


def mnist_exp_cw(ROOT_PATH, baselines):
    executor_list = []
    for eps in [2.0]:
        for conf in [1, 5, 10, 50, 100, 200, 400]:
            for lr in [0.01, 0.1, 0.2]:
                attack = "cw"
                attack_args = CWParam(
                    "mnist", eps, lr, 1000, 100, 9, conf, attack_size=1000)
                EXP_PATH = "mnist_{}_{}_{}_{}_{}_{}".format(attack,
                                                            "targeted" if "targeted" in attack_args.keys() else "untargeted",
                                                            attack_args["eps_type"],
                                                            attack_args["eps"], conf, lr)

                for bl in baselines:
                    print("Baseline: ", bl)
                    bex = BaselineExecutor(
                        bl, attack_args, os.path.join(ROOT_PATH, EXP_PATH))
                    executor_list.append(bex)

    return executor_list


def mnist_exp():
    ROOT_PATH = "./results/mnist_advanced/"
    baselines = ["arclu"]

    executor_list = []
    executor_list.extend(mnist_exp_auto(ROOT_PATH, baselines))
    executor_list.extend(mnist_exp_random_start(ROOT_PATH, baselines))
    executor_list.extend(mnist_exp_multi_targeted(ROOT_PATH, baselines))
    executor_list.extend(mnist_exp_knn(ROOT_PATH, ["arclu"]))
    executor_list.extend(mnist_exp_cw(ROOT_PATH, ["arclu"]))
    print(list(range(torch.cuda.device_count())))
    Scheduler(list(range(torch.cuda.device_count())), executor_list)


if __name__ == "__main__":
    mnist_exp()
