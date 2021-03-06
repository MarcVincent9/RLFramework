# -*- coding: utf-8 -*-

import signal
import inspect
import math
import numpy as np
from timeit import default_timer as timer


def roulette(distribution, size=None, replace=True, shuffle=False):
    """
    randomly return one of the dictionary's key according to its probability (its value).
    Wrapper for numpy's choice method to work with dictionaries.

    :param distribution: dictionary {index: probability}
    :param size: int
    :rtype: index (or ndarray if size specified)
    """
    return np.random.default_rng().choice(list(distribution.keys()), size, 
                                          replace, p=list(distribution.values()),
                                          shuffle=shuffle)
    

def timeout(func):
    """Decorator. When applied to a method which has a "time_limit" keyword, limits the method's runtime to
    the specified duration. Always records the total runtime as an attribute "time" of the method's object."""

    def handler(signum, frame):
        raise TimeoutError()

    def new_func(self, *args, **kwargs):
        start_time = timer()
        try:
            old = signal.signal(signal.SIGALRM, handler)
            try:
                time_limit = kwargs["time_limit"]
            except KeyError:
                time_limit = inspect.signature(func).parameters["time_limit"].default
            signal.alarm(math.ceil(time_limit))
            try:
                result = func(self, *args, **kwargs)
            except TimeoutError:
                print("Time out")
                result = None
            finally:
                signal.signal(signal.SIGALRM, old)
            signal.alarm(0)
            return result
        except KeyError:
            return func(self, *args, **kwargs)
        finally:
            self.time = timer() - start_time
    new_func.__name__ = func.__name__
    return new_func
