import time


class Timer(object):
    def __init__(self, verbose=True):
        self.verbose = verbose

    def __call__(self, signature, func, args=(), kwargs={}):
        begin = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        dur = round(end - begin, 3)
        if self.verbose:
            print("{} seconds to execute {} ".format(dur, signature))
        return result
