import sys
import os
import subprocess
from queue import Queue
import threading
from glob import glob


class Scheduler(object):
    def __init__(self, gpu_list, executors_list):
        self.waiting_queue = Queue()
        self.gpu_queue = Queue()
        # threads notify Scheduler they finish their job by queing ready_queue
        self.ready_queue = Queue()

        for i in gpu_list:
            self.gpu_queue.put(i)

        for e in executors_list:
            self.waiting_queue.put(e)

        xs = []
        for i in range(min(len(gpu_list), self.waiting_queue.qsize())):
            e = self.waiting_queue.get()
            x = threading.Thread(target=self.start_executor, args=(e,))
            xs.append(x)

        for x in xs:
            x.start()

        self.schedule()

    def schedule(self):
        while self.ready_queue.empty() == False or self.waiting_queue.empty() == False:
            ready_e = self.ready_queue.get(block=True)

            x = threading.Thread(target=self.start_executor, args=(ready_e,))
            x.start()

    def start_executor(self, e):
        print("----start executor", e.directory, self.gpu_queue.qsize())
        print("----start executor", e.directory)
        e.set_gpu(self.gpu_queue.get())

        try:
            e.execute()
        except Exception:
            print("execution exception occured!")

        self.gpu_queue.put(e.get_gpu())
        print("end executor", e.directory)
        if self.waiting_queue.empty() == False:
            self.ready_queue.put(self.waiting_queue.get())


class Executor():
    def __init__(self, directory, cmd):
        self.directory = directory
        self.cmd = cmd
        self.gpu_num = 0

    def set_gpu(self, gpu_num):
        self.gpu_num = gpu_num

    def get_gpu(self):
        return self.gpu_num

    def execute(self):
        self.before_script()
        print("Execute:", self.cmd)
        self.run_script()
        self.after_script()

    def before_script(self):
        pass

    def run_script(self):
        os.system(self.cmd)

    def after_script(self):
        pass


class BaselineExecutor(Executor):
    def __init__(self, baseline_name, attack_args, result_dir):
        self.gpu_num = 0
        self.FILE_DIR = os.path.dirname(__file__)
        self.baseline_name = baseline_name
        self.attack_args = attack_args.copy()
        self.DIRS = {"arclu": os.path.join(self.FILE_DIR, "../../../arclu/"),
                     }
        self.BASE_CMDS = {"arclu": "python test.py --batch_size 20",
                          }

        self.ENV_NAMES = {"arclu": "envtest_arclu",
                          }

        start_dir = os.path.abspath(os.getcwd())
        self.result_full_path = os.path.join(
            start_dir, result_dir, baseline_name)

        self.directory = self.DIRS[self.baseline_name]
        self.cmd = self.make_cmd()

    def before_script(self):
        if not os.path.exists(self.result_full_path):
            os.makedirs(self.result_full_path, exist_ok=True)

    def set_gpu(self, gpu_num):
        self.gpu_num = gpu_num
        self.cmd = self.make_cmd()

    def make_cmd(self):
        env_activate_script = "eval \"$(conda shell.bash hook)\"; \
                                conda activate {}".format(self.ENV_NAMES[self.baseline_name])

        gpu_script = "CUDA_VISIBLE_DEVICES={}".format(self.gpu_num)

        change_dir_sctr = "cd " + self.directory

        attack_param_str = " ".join(
            ["--{} {}".format(k, v) for k, v in self.attack_args.items()])

        return change_dir_sctr + "; " + env_activate_script + "; "\
            + gpu_script + " " + self.BASE_CMDS[self.baseline_name] + " " + attack_param_str + \
            " --result_path " + self.result_full_path + "/"

    def run_script(self):
        if self.baseline_name != "resisting" and self.baseline_name != "resisting_cifar":
            super(BaselineExecutor, self).run_script()
        else:
            out = str(subprocess.check_output(self.cmd, shell=True))
            result_txt = "\n".join(out.split("PRINT FOR COPY")[
                                   1].split("\\n")[1:9])
            txt_file = open(os.path.join(
                self.result_full_path, "result.txt"), "w+")
            txt_file.write(result_txt)
            txt_file.close()
            print(result_txt)

    def after_script(self):
        pass


def PGDParam(dataset, eps_type, eps, step_size, iters, attack_size=10000):
    attack_args = {}
    attack_args["dataset"] = dataset
    attack_args["attack"] = "pgd"
    attack_args["eps_type"] = str(eps_type)
    attack_args["eps"] = str(eps)
    attack_args["step_size"] = str(step_size)
    attack_args["iters"] = str(iters)
    attack_args["attack_size"] = str(attack_size)
    return attack_args


def PGDDisperseParam(dataset, eps_type, eps, step_size, iters, zero_threshold_multiplier, adv_coeff, attack_size=10000):
    attack_args = {}
    attack_args["dataset"] = dataset
    attack_args["attack"] = "pgddisperse"
    attack_args["eps_type"] = str(eps_type)
    attack_args["eps"] = str(eps)
    attack_args["step_size"] = str(step_size)
    attack_args["iters"] = str(iters)
    attack_args["attack_size"] = str(attack_size)

    attack_args["zero_threshold_multiplier"] = str(zero_threshold_multiplier)
    attack_args["adv_coeff"] = str(adv_coeff)

    return attack_args


def MultiTargetDisperseParam(dataset, eps_type, eps, step_size, iters, num_classes, zero_threshold_multiplier, adv_coeff, attack_size=10000):
    attack_args = {}
    attack_args["dataset"] = dataset
    attack_args["attack"] = "pgddisperse"
    attack_args["eps_type"] = str(eps_type)
    attack_args["eps"] = str(eps)
    attack_args["step_size"] = str(step_size)
    attack_args["iters"] = str(iters)
    attack_args["attack_size"] = str(attack_size)
    attack_args["targeted"] = ""
    attack_args["multi_targeted"] = ""
    # Repetition should be "num_classes"
    attack_args["repetition"] = num_classes

    attack_args["zero_threshold_multiplier"] = str(zero_threshold_multiplier)
    attack_args["adv_coeff"] = str(adv_coeff)

    return attack_args


def NESParam(dataset, eps_type, eps, step_size, iters, n_sample, search_sigma, attack_size=10000):
    attack_args = {}
    attack_args["dataset"] = dataset
    attack_args["attack"] = "nes"
    attack_args["eps_type"] = str(eps_type)
    attack_args["eps"] = str(eps)
    attack_args["step_size"] = str(step_size)
    attack_args["iters"] = str(iters)
    attack_args["n_sample"] = str(n_sample)
    attack_args["search_sigma"] = str(search_sigma)
    attack_args["attack_size"] = str(attack_size)
    return attack_args


def BoundaryParam(dataset, eps_type, eps, n_delta, init_delta, init_epsilon, iters, attack_size=10000):
    attack_args = {}
    attack_args["dataset"] = dataset
    attack_args["attack"] = "boundary"
    attack_args["eps_type"] = str(eps_type)
    attack_args["eps"] = str(eps)
    attack_args["n_delta"] = str(n_delta)
    attack_args["init_delta"] = str(init_delta)
    attack_args["init_epsilon"] = str(init_epsilon)
    attack_args["iters"] = str(iters)
    attack_args["attack_size"] = str(attack_size)
    return attack_args


def TransferParam(dataset):
    attack_args = {}
    attack_args["dataset"] = dataset
    attack_args["attack"] = "transfer"
    return attack_args


def SemanticParam(dataset, eps_type, eps, step_size, iters, attack_size=10000):
    attack_args = {}
    attack_args["dataset"] = dataset
    attack_args["attack"] = "semantic"
    attack_args["eps_type"] = str(eps_type)
    attack_args["eps"] = str(eps)
    attack_args["step_size"] = str(step_size)
    attack_args["iters"] = str(iters)
    attack_args["attack_size"] = str(attack_size)
    return attack_args


def FGSMParam(dataset, eps, attack_size=10000):
    attack_args = {}
    attack_args["dataset"] = dataset
    attack_args["attack"] = "fgsm"
    attack_args["eps_type"] = "linf"
    attack_args["eps"] = str(eps)
    attack_args["step_size"] = -9999  # dummy for compatibility
    attack_args["iters"] = -9999  # dummy for compatibility
    attack_args["attack_size"] = str(attack_size)
    return attack_args


def MIMParam(dataset, eps_type, eps, step_size, iters, attack_size=10000):
    attack_args = {}
    attack_args["dataset"] = dataset
    attack_args["attack"] = "mim"
    attack_args["eps_type"] = str(eps_type)
    attack_args["eps"] = str(eps)
    attack_args["step_size"] = str(step_size)
    attack_args["iters"] = str(iters)
    attack_args["attack_size"] = str(attack_size)
    return attack_args


def CWParam(dataset, eps, step_size, iters, initial_const, binary_search_steps, confidence, attack_size=10000):
    attack_args = {}
    attack_args["dataset"] = dataset
    attack_args["attack"] = "cw"
    attack_args["eps_type"] = "l2"
    attack_args["eps"] = str(eps)
    attack_args["step_size"] = str(step_size)
    attack_args["iters"] = str(iters)

    attack_args["initial_const"] = str(initial_const)
    attack_args["binary_search_steps"] = str(binary_search_steps)
    attack_args["confidence"] = str(confidence)

    attack_args["attack_size"] = str(attack_size)
    return attack_args


def AutoParam(dataset, eps_type, eps, attack_size=10000):
    attack_args = {}
    attack_args["dataset"] = dataset
    attack_args["attack"] = "auto"
    attack_args["eps_type"] = eps_type
    attack_args["eps"] = str(eps)
    attack_args["step_size"] = -1  # str(step_size)

    attack_args["attack_size"] = str(attack_size)
    return attack_args


def SquareParam(dataset, eps, attack_size=10000):
    attack_args = {}
    attack_args["dataset"] = dataset
    attack_args["attack"] = "square"
    attack_args["eps_type"] = "linf"
    attack_args["eps"] = str(eps)
    attack_args["step_size"] = -1  # str(step_size)

    attack_args["attack_size"] = str(attack_size)
    return attack_args


def FABtParam(dataset, eps, attack_size=10000):
    attack_args = {}
    attack_args["dataset"] = dataset
    attack_args["attack"] = "fabt"
    attack_args["eps_type"] = "linf"
    attack_args["eps"] = str(eps)
    attack_args["step_size"] = -1  # str(step_size)

    attack_args["attack_size"] = str(attack_size)
    return attack_args


def APGDtParam(dataset, eps, attack_size=10000):
    attack_args = {}
    attack_args["dataset"] = dataset
    attack_args["attack"] = "apgdt"
    attack_args["eps_type"] = "linf"
    attack_args["eps"] = str(eps)
    attack_args["step_size"] = -1  # str(step_size)

    attack_args["attack_size"] = str(attack_size)
    return attack_args


def KNNParam(dataset, eps_type, eps, step_size, iters, attack_size=10000):
    attack_args = {}
    attack_args["dataset"] = dataset
    attack_args["attack"] = "knn"
    attack_args["eps_type"] = str(eps_type)
    attack_args["eps"] = str(eps)
    attack_args["step_size"] = str(step_size)
    attack_args["iters"] = str(iters)
    attack_args["attack_size"] = str(attack_size)
    attack_args["targeted"] = ""
    return attack_args


def RandomStartParam(dataset, eps_type, eps, step_size, iters, repetition, attack_size=10000):
    attack_args = {}
    attack_args["dataset"] = dataset
    attack_args["attack"] = "pgd"
    attack_args["eps_type"] = str(eps_type)
    attack_args["eps"] = str(eps)
    attack_args["step_size"] = str(step_size)
    attack_args["iters"] = str(iters)
    attack_args["attack_size"] = str(attack_size)
    attack_args["random_start"] = ""
    attack_args["repetition"] = str(repetition)

    return attack_args


def MultiTargetParam(dataset, eps_type, eps, step_size, iters, num_classes, attack_size=10000):
    attack_args = {}
    attack_args["dataset"] = dataset
    attack_args["attack"] = "pgd"
    attack_args["eps_type"] = str(eps_type)
    attack_args["eps"] = str(eps)
    attack_args["step_size"] = str(step_size)
    attack_args["iters"] = str(iters)
    attack_args["attack_size"] = str(attack_size)
    attack_args["targeted"] = ""
    attack_args["multi_targeted"] = ""
    # Repetition should be "num_classes"
    attack_args["repetition"] = num_classes

    return attack_args
