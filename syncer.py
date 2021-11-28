import subprocess
import sys
import time
import os
args = sys.argv

root = "~/dprk-research/experiment_outputs"
source_folder = os.path.join(root,args[1],"plots")
save_folder = os.path.join("../experiment_outputs",args[1])
if not os.path.exists(save_folder):os.makedirs(save_folder)

cmd = "scp -r hgc:{}/{}*/plots {}".format(root,args[1],save_folder)
subprocess.call(cmd,shell=True)
time.sleep(5)