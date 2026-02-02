import datetime

class OrderController():

    def __init__(self):
        self._orders = list()
        self._typeAction = None
    
    def choose(self, action):    
        self._typeAction = action

        match(action):
            case 1: 
                return self.addOrder()
            case 2: self.removeOrder()
            case 3: self.searchOrder()
            case 4: self.showOrders()
            case _: return False # любое значение кроме 1,2,3,4
        return True



    def showOrders(self):
        for order in self._orders:
            print(order)

    def addOrder(self):
        
        cost = None
        name = input("Name: ")
        
        try:
            cost = int(input("Cost: "))
            typeName = input("Type: ")

            self._orders.append(Order(name, cost, typeName))
        except ValueError:
            print("Ожидалось число")
        
        return True

    def removeOrder(self):
        # удаление по имени(Name)
        searchString = input("search:")
        for order in self._orders:
            if order.getName() == searchString:
                self._orders.remove(order)
                break
        
    def searchOrder(self):
        searchString = input("search:")
        for order in self._orders:
            if order.getName() == searchString:
                print(order)
                break

class Order():

    def __init__(self, name, cost, typeName):
        self._name = name
        self._cost = cost
        self._type = typeName
        self._ts = datetime.datetime.now()

    def getName(self):
        return self._name

    def __repr__(self):
        return f'Type: {self._type}, Name: {self._name}, Cost: {self._cost}, TS: {self._ts}'
    
