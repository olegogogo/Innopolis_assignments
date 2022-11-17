from datetime import datetime


def fib(n):
    if n < 2:
        return n
    return fib(n - 2) + fib(n - 1)


class Some:
    num_of_instances = 6

    def __init__(self):
        Some.num_of_instances += 1
        self.serial = hex(fib(self.num_of_instances))
        self.time = datetime.now()

    def SerialNumber(self):
        return 'I am object number {}'.format(self.serial)

    def __str__(self):
        return 'I am object number {}'.format(self.serial), 'with creation date: {}'.format(self.time)


if __name__ == '__main__':
    obj1 = Some()
    obj2 = Some()
    obj3 = Some()

    print(obj1.SerialNumber())
    print(obj2.SerialNumber())
    print(obj3.SerialNumber())