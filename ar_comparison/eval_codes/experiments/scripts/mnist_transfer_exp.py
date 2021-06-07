import os
import sys
import torch
sys.path.append("./")
from execution import TransferParam, BaselineExecutor, Scheduler


def mnist_exp_semantic_linf(ROOT_PATH, baselines):
    executor_list = []
    EXP_PATH = ""

    attack_args = TransferParam("mnist")

    executor_list = []
    for bl in baselines:
        print("Baseline: ", bl)
        bex = BaselineExecutor(
            bl, attack_args, os.path.join(ROOT_PATH, EXP_PATH))
        executor_list.append(bex)

    return executor_list


def mnist_exp():
    ROOT_PATH = "./results/mnist_transfer/"
    baselines = ["arclu"]

    executor_list = []
    executor_list.extend(mnist_exp_semantic_linf(ROOT_PATH, baselines))
    Scheduler(list(range(torch.cuda.device_count())), executor_list)


if __name__ == "__main__":
    mnist_exp()
