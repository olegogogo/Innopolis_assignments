class Int:
    def __init__(self, value=0):
        self.value = value

    def __add__(self, a):
        return Int(self.value + a.value)

    def __str__(self):
        return str(self.value)


if __name__ == "__main__":
    a = Int()
    b = Int(2)
    c = Int(3)
    a = b + c
    print(a)
    print(type(a))
