import subprocess
import sys
import time
import os



def sync_single(args):
    # for x in ["0-{}".format(str(a)) for a in range(1,10)]+['1']:
    root = "~/dprk-research/experiment_outputs"
    # source_folder = os.path.join(root,args[1]+ '_'+x,"plots")
    source_folder = os.path.join(root,args[1],"plots")
    save_folder = os.path.join("../experiment_outputs",args[1])
    if not os.path.exists(save_folder):os.makedirs(save_folder)
    cmd = "scp -r hgc:{} {}".format(source_folder,save_folder)
    subprocess.call(cmd,shell=True)

def sync_hyperparam(args):
    # for x in ["0-{}".format(str(a)) for a in range(1,10)]+['1']:
    root = "~/dprk-research/experiment_outputs"
    # source_folder = os.path.join(root,args[1]+ '_'+x,"plots")
    source_folder = os.path.join(root, args[1], "best_model_folder")
    save_folder = os.path.join("../experiment_outputs", args[1])
    if not os.path.exists(save_folder): os.makedirs(save_folder)
    cmd = "scp -r hgc:{}/* {}".format(source_folder, save_folder)
    subprocess.call(cmd, shell=True)


def main():
    function_map = {"single":sync_single,
                    "hyper":sync_hyperparam}
    args = sys.argv
    mode = args[2]
    print("Mode",mode)
    if mode not in function_map.keys():
        raise ValueError("Unknown mode {}".format(mode))
    else:
        function_map[mode](args)


if __name__ == "__main__":
    main()
