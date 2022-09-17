import datetime
import time
import traceback
from inspect import signature, getsource
from task1 import blockPrint

def decorator_fun_error(func):
    countcall = dict()

    def timed(*args, **kwargs):
        try:
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
        except Exception as err:
            with open('exeptions.log', 'a') as file:
                file.write('Exeption happened at {}\n'.format(str(datetime.datetime.now())))
                file.write('{}\n'.format(traceback.format_exc()))
    return timed

class decorator_class_error:
    global countcall
    countcall = dict()

    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        try:
            timestart = time.time()
            result = blockPrint(self.func, *args, **kwargs)
            timefinish = time.time()
            if self.func.__name__ in countcall:
                countcall[self.func.__name__] += 1
            else:
                countcall[self.func.__name__] = 1
            with open("_output.txt", 'a') as file:
                file.write('Function {} call, {} : time: {}, ms\n'.format(self.func.__name__, countcall[self.func.__name__],
                                                                          (timefinish - timestart) * 1000))
            return result
        except Exception as err:
            with open('exeptions.log', 'a') as file:
                file.write('Exeption happened at {}\n'.format(str(datetime.datetime.now())))
                file.write('{}\n'.format(traceback.format_exc()))