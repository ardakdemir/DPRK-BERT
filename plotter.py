import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


class BasicPlotter:
    """
        update all plots with a single command
    """
    def __init__(self,save_root_folder,prefix="",metrics = {}):
        self.save_root = save_root_folder
        self.metrics = defaultdict(list)
        self.prefix = prefix
        if not os.path.exists(self.save_root):
            os.makedirs(self.save_root)

    def send_metrics(self,metrics):
        old_metrics = self.metrics
        for k,v in metrics.items():
            old_metrics[k].append(v)

        for k,v in old_metrics.items():
            p = os.path.join(self.save_root,self.prefix + k+".png")

            plt.figure()
            plt.plot(v)
            plt.ylabel(k)
            plt.xticks(np.arange(0, len(v) + 1, 1.0))
            plt.savefig(p)
            plt.close("all")
