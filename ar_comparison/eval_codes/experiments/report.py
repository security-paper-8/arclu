import sys
import os
import subprocess
from glob import glob


def run(cmd):
    proc = subprocess.Popen(cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            shell=True
                            )
    stdout, stderr = proc.communicate()

    return proc.returncode, stdout, stderr


class Reporter(object):
    def __init__(self, directory, FPR):
        self.dir = directory
        self.FPR = FPR

    def report(self, raw_output=False):
        report_path = os.path.join(self.dir, "**/result_*.npz")

        dirs = glob(report_path)
        cmds = [
            "python ../test.py {} {}|grep \"PRINT FOR\" -A 7".format(d, self.FPR) for d in dirs]
        cmds.extend(["cat {}".format(d) for d in glob(os.path.join(
            self.dir, "**/*.txt")) if "resisting" in d or "resisting_cifar" in d])
        if raw_output:
            [os.system(c) for c in cmds]
        else:
            # Parse outputs
            str_outputs = [str(run(c)[1]) for c in cmds]
            print(len(str_outputs))
            dict_out = {}

            for o in str_outputs:
                numbers = []
                baseline_name = None
                for l in o.split("\\n"):
                    ll = l.replace('b\'', "").replace('\'', "")
                    if "PRINT" in l or l.startswith("results"):
                        baseline_name = l.split("/")[-2].split("_")[0]
                    else:

                        try:
                            numbers.append(float(ll))
                        except ValueError:
                            continue
                if baseline_name != "resisting":
                    dict_out[baseline_name] = {"ASR": "{:.04f}".format(
                        numbers[0]), "EROC": "{:0.04f}".format(numbers[-1])}
                else:
                    dict_out[baseline_name] = {"ASR": "{:.04f}".format(
                        numbers[0]), "EROC": "{:0.04f},{:0.04f}".format(numbers[-2], numbers[-1])}

            for b in ["arclu"]:
                if b in dict_out.keys():
                    print("{},\t{},\t{}".format(
                        b, dict_out[b]["ASR"], dict_out[b]["EROC"]))

        return dirs


if __name__ == "__main__":
    di = sys.argv[1]
    FPR = sys.argv[2] if len(sys.argv) == 3 else 5
    print(di, FPR)
    Reporter(di, FPR).report()
