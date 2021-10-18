import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def joint_plot(value_dict, save_path, y_label=None):
    root = os.path.split(save_path)[0]
    if not os.path.exists(root):os.makedirs(root)
    plt.figure(figsize=(12, 8))
    for c, n in value_dict.items():
        plt.plot(n, label=c)
    if y_label is not None:
        plt.ylabel(y_label)
    plt.xticks(np.arange(0, len(list(value_dict.values())[0]), 1.0))
    plt.legend()
    plt.savefig(save_path)
    plt.close("all")


class BasicPlotter:
    """
        update all plots with a single command
    """

    def __init__(self, save_root_folder, prefix="", metrics={}):
        self.save_root = save_root_folder
        self.metrics = defaultdict(list)
        self.prefix = prefix
        if not os.path.exists(self.save_root):
            os.makedirs(self.save_root)

    def send_metrics(self, metrics):
        old_metrics = self.metrics
        for k, v in metrics.items():
            old_metrics[k].append(v)

        for k, v in old_metrics.items():
            p = os.path.join(self.save_root, self.prefix + k + ".png")

            plt.figure(figsize=(12, 8))
            plt.plot(v)
            plt.ylabel(k)
            plt.xticks(np.arange(0, len(v) + 1, 1.0))
            plt.savefig(p)
            plt.close("all")


def main():
    charts = [[8, 8, 3, 3, 4, 4], [1, 1, 2, 2, 1, 1]]
    names = ["kr-bert", "dprk-bert"]
    y_label = "perplexity"
    save_path = "../experiment_outputs/dummy_plot.png"
    joint_plot(charts, names, save_path, y_label)


if __name__ == "__main__":
    main()
