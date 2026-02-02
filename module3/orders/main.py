from module3.orders.order import OrderController

def menu():
    print(f"1. add\n2. remove\n3. search\n4. show\n5. exit")

if __name__ == "__main__":
    try:
        controller = OrderController()
        is_run = True   

        while(is_run):

            menu()

            is_run = controller.choose(int(input(">")))
    except KeyboardInterrupt: # нормальное завершение работы консольной программы
        print("Завершение работы")

