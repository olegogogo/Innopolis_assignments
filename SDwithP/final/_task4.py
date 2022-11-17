class employee:
    data = {'loleg': 1, 'peotr': 2, 'meme': 3}
    def find(self):
        return print(self.data.get(input()))

if __name__ == '__main__':
    a = employee()
    a.find()