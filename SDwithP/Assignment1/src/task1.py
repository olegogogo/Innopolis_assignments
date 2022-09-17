import os
import random
import sys
import time


def blockPrint(func, *args, **kwargs):
    sys.stdout = open(os.devnull, 'w')  # blocking print
    result = func(*args, **kwargs)
    time.sleep(random.uniform(0.0, 0.1))  # just to be clear that all correct
    sys.stdout = sys.__stdout__  # unblock printing
    return result


def decorator0(func):
    countcall = dict()

    def timed(*args, **kwargs):
        timestart = time.time()
        result = blockPrint(func, *args, **kwargs)
        timefinish = time.time()
        if func.__name__ in countcall:
            countcall[func.__name__] += 1
        else:
            countcall[func.__name__] = 1
        print('Function', func.__name__, 'call', countcall[func.__name__], ': time:', (timefinish - timestart) * 1000,
              'ms')
        return result

    return timed
