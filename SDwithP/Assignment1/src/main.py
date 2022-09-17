from task1 import decorator0
from task2 import decorator1
from task3 import decorator21, decorator22, fun1, fun2, fun3, fun4, ranking
from task4 import decorator_fun_error, decorator_class_error
from math import sqrt


@decorator21  # calculate execution time and print in _output.txt file
def mul_num(*args):
    l = list(*args)
    return list(map(lambda x: x ** 2, l))


@decorator22  # print all information about function to _output.txt file
def inc_num(*args):
    l = list(*args)
    return list(map(lambda x: x + 1, l))


@decorator0  # calculate execution time and print in console
def quad(a=10, b=5, c=2):
    D = b ** 2 - 4 * a * c
    if D >= 0:
        x1 = (-b + sqrt(D)) / 2 * a
        x2 = (-b - sqrt(D)) / 2 * a
    if D >= 0:
        print("with numbers {}, x1 = {}, x2 = {}".format((a, b, c), x1, x2))
    else:
        print("there is no solution")


@decorator0
def pascal(n=5):
    row = [1]
    for i in range(5):
        print(row)
        row = [sum(j) for j in zip([0] + row, row + [0])]


@decorator1  # print all information about function to console
def funh(bar1, bar2=""):
    """
    This function does something useful
    :param bar1: description
    :param bar2: description
    """
    print("some\nmultiline\noutput")


@decorator_fun_error
def error(a='hello', b=2):  # for task 4
    return a + b

@decorator_class_error
def error2(a='bye', b=5):
    return a - b

print('Task 1:')
pascal()
quad()
pascal()
quad()
pascal()
print()

print('Task 2:')
funh(None, bar2="")
print()

print('Task 3:')
mul_num([1, 2, 3])
inc_num([1, 2, 3])
print()
ranking(fun1, fun2, fun3, fun4)
print()

print('Task 4:')
print('There are an errors!')
error()
error2()
