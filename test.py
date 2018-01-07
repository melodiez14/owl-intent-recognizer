class Testing:
    def __init__(self):
        self.__test__ = 2

    def get_names(self):
        return self.__hash__()

if __name__ == '__main__':
    t = Testing()
    print(t.get_names())