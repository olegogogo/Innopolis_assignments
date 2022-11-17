class date:
    def getDate(self):
        month, day, year = input().split('/')
        self.month = month
        self.day = day
        self.year = year

    def showDate(self):
        return print(self.month, self.day, self.year)

if __name__ == '__main__':
    a = date()
    b = date()
    a.getDate()
    b.getDate()
    a.showDate()
    b.showDate()