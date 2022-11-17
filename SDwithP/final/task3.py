import task1, task2

class Ship:
    num_of_instances = 0
    instance = None
    def __new__(self, *args, **kwargs):
        if Ship.num_of_instances < 3:
            self.instance = super().__new__(self)
            Ship.num_of_instances += 1
        else:
            Ship.num_of_instances = 3
        return self.instance


    def __init__(self):
        self.serial = self.num_of_instances
        self.name = 'Ship' + str(self.num_of_instances)
        self.capacity = 'some random value'
        self.lattitudeInst = task1.Angle.inputting()
        self.lattitude = self.lattitudeInst.__str__()
        self.longtitudeInst = task1.Angle.inputting()
        self.longtitude = self.longtitudeInst.__str__()

    def __str__(self):
        return 'Ship name: {}, number: {}, location: {} latitude, and {} longitude, capacity: {}'.format(self.name, self.serial, self.lattitude, self.longtitude, self.capacity)


if __name__ == '__main__':
    ship1 = Ship()
    ship2 = Ship()
    # ship3 = Ship()
    # ship4 = Ship()

    print(ship1.__str__())
    print(ship2.__str__())
    # print(ship3.__str__())
    # print(ship4.__str__())