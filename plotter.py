import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import json


def joint_line_plot(value_dict, save_path, y_label=None):
    root = os.path.split(save_path)[0]
    if not os.path.exists(root): os.makedirs(root)
    plt.figure(figsize=(12, 8))
    for c, n in value_dict.items():
        plt.plot(n, label=c)
    if y_label is not None:
        plt.ylabel(y_label)

    plt.xticks(np.arange(0, len(list(value_dict.values())[0]), 1.0))
    plt.legend()
    plt.savefig(save_path)
    plt.close("all")


def joint_bar_plot(value_dict, save_path, y_label=None, width=0.25, separation=0.5):
    """
        Cool script to auto-adjust the bar separations
        Let me create a plotting repository for this
    """
    root = os.path.split(save_path)[0]
    if not os.path.exists(root): os.makedirs(root)
    plt.figure(figsize=(12, 8))
    model_names = list(value_dict.keys())
    num = len(model_names)
    offset = lambda index: 1 + separation * index + (num * width) * index
    x_positions = [offset(index) for index in range(len(value_dict[model_names[0]]))]
    for i, m in enumerate(model_names):
        values = value_dict[m]
        plt.bar(x=[offset(index) + width * i - ((num - 1) / 2) * width for index in range(len(values))], height=values,
                label=m, width=width)
    if y_label is not None:
        plt.ylabel(y_label)
    plt.xticks(x_positions, [str(1 + i) for i in range(len(x_positions))])
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
        self.max_xticks = 50
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
            # plt.xticks(np.arange(1,len(v)+1,max(1,len(v)//self.max_xticks)),)
            plt.savefig(p)
            plt.close("all")

    def store_json(self,additional_data={}):
        p = os.path.join(self.save_root, self.prefix + "training_metrics" + ".json")
        with open(p, "w") as j:
            self.metrics.update(additional_data)
            json.dump(self.metrics, j)


def main():
    charts = [[8, 8, 3, 3, 4, 4], [1, 1, 2, 2, 1, 1]]
    names = ["kr-bert", "dprk-bert"]
    y_label = "perplexity"
    save_path = "../experiment_outputs/dummy_plot.png"
    joint_plot(charts, names, save_path, y_label)


if __name__ == "__main__":
    main()
