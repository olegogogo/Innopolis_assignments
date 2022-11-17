class time:
    def __init__(self, hours=0, minutes=0, seconds=0):
        self.hours = hours
        self.minutes = minutes
        self.seconds = seconds

    def display(self):
        return f'{self.hours}:{self.minutes}:{self.seconds}'

    def __add__(self, other):
        return time(self.hours + other.hours, self.minutes + other.minutes, self.seconds + other.seconds)


if __name__ == '__main__':
    a = time()
    b = time(1,1,1)
    c = time(2,2,2)
    a = b+c
    print(a.display())
