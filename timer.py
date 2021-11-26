import time
from collections import defaultdict
import json

class Timer(object):
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.cache = defaultdict(list)

    def __call__(self, signature, func, args=(), kwargs={}):
        begin = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        dur = round(end - begin, 3)
        self.update(signature, dur)
        if self.verbose:
            print("{} seconds to execute {} ".format(dur, signature))
        return result

    def update(self, signature, dur):
        self.cache[signature].append(dur)

    def report(self):
        for k,v in self.cache.items():
            print("Average time (after {} data points) for {}: {} secs".format(len(v),k,sum(v)/len(v)))

    def store_json(self,save_path):
        with open(save_path,"w") as o:
            json.dump(self.cache,o)

