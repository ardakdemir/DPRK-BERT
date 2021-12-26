import subprocess
import sys
import time
import os
args = sys.argv


# for x in ["0-{}".format(str(a)) for a in range(1,10)]+['1']:
root = "~/dprk-research/experiment_outputs"
# source_folder = os.path.join(root,args[1]+ '_'+x,"plots")
source_folder = os.path.join(root,args[1],"plots")
save_folder = os.path.join("../experiment_outputs",args[1])
if not os.path.exists(save_folder):os.makedirs(save_folder)
cmd = "scp -r hgc:{} {}".format(source_folder,save_folder)
subprocess.call(cmd,shell=True)