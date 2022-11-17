from datetime import datetime
import keyboard


class toolBooth:
    numCars = 0
    money = 0

    def decorator(func):
        countCall = dict()

        def decor(*args, **kwargs):
            if func.__name__ in countCall:
                countCall[func.__name__] += 1
            else:
                countCall[func.__name__] = 1

            with open('logfile.txt', 'a') as file:
                file.write(f"Function, {func.__name__}, call, {countCall[func.__name__]}, : time:, {datetime.now()} ")
            return func(*args, **kwargs)

        return decor

    @decorator
    def payingCar(self):
        toolBooth.numCars += 1
        toolBooth.money += 0.5

    @decorator
    def nopayingCar(self):
        toolBooth.numCars += 1

    @decorator
    def display(self):
        return toolBooth.numCars, toolBooth.money


if __name__ == '__main__':
    a = toolBooth()
    while True:
        if keyboard.read_key() == 'q':
            a.payingCar()
            print('car passed')
        if keyboard.read_key() == 'w':
            a.payingCar()
            print('car passed -')
        if keyboard.read_key() == 'esc':
            print(a.display())
            exit()
