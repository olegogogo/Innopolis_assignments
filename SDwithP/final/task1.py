class Angle:
    def __init__(self, degree: int, minutes: float, direction: str):
        self.degree = degree
        self.minutes = minutes
        self.direction = direction

    def inputting():
        degree = int(input())
        minutes = float(input())
        direction = input()
        return Angle(degree, minutes, direction)

    def __str__(self):
        return str(self.degree) + u'\N{DEGREE SIGN}' + str(self.minutes) + "'" + self.direction


if __name__ == '__main__':
    angle = Angle(179, 59.9, 'E')
    print(angle.__str__())
    while True:
        angle = Angle.inputting()
        print(angle.__str__())