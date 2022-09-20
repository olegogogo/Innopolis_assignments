import random
import time
from inspect import getsource, signature

from task1 import blockPrint, decorator0


class decorator21:
    global countcall
    countcall = dict()

    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        timestart = time.time()
        result = blockPrint(self.func, *args, **kwargs)
        timefinish = time.time()
        if self.func.__name__ in countcall:
            countcall[self.func.__name__] += 1
        else:
            countcall[self.func.__name__] = 1
        with open("_output.txt", "a") as file:
            file.write(
                "Function {} call, {} : time: {}, ms\n".format(
                    self.func.__name__,
                    countcall[self.func.__name__],
                    (timefinish - timestart) * 1000,
                )
            )
        return result


class decorator22:
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        result = decorator0(self.func)(*args, **kwargs)
        with open("_output.txt", "a") as file:
            file.write("Name: {}\n".format(self.func.__name__))
            file.write("Type: {}\n".format(str(type(self.func))))
            file.write("Sign: {}\n".format(str(signature(self.func))))
            file.write("Args: {} positional\n".format(str(args)))
            file.write("Args: {} keyworded\n".format(str(kwargs)))
            file.write("\n")
            file.write("Doc: {}\n".format(self.func.__doc__))
            file.write("\n")
            file.write("Source: {}\n".format(getsource(self.func)))
            file.write("\n")
            file.write("Output: \n")
            file.write(str(self.func(*args)))
        return result


# some small functions to finish task 3
def fun1():
    time.sleep(random.uniform(0.0, 0.1))


def fun2():
    time.sleep(random.uniform(0.0, 0.1))


def fun3():
    time.sleep(random.uniform(0.0, 0.1))


def fun4():
    time.sleep(random.uniform(0.0, 0.1))


def ranking(*args):
    rank = {}
    for i in args:
        timestart = time.time()
        i()
        timefinish = time.time()
        rank[i.__name__] = (timefinish - timestart) * 1000
    rank = sorted(rank.items(), key=lambda item: item[1])
    print("{:<10}|{:<10}|{:<10}".format(*["PROGRAM", "RANK", "TIME ELAPSED"]))
    temp = []
    for name, times in rank:
        temp.append(name)
        print("{:<10}|{:<10}|{:<10.10f} ms".format(name, temp.index(name) + 1, times))
