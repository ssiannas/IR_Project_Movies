from src import search
import os

def cls():
    os.system('cls' if os.name=='nt' else 'clear')

# now, to clear the screen

if __name__ == "__main__":
    while True:
        cls()
        print("Chose one of the following options:")
        my_search = search.Search()
        choice = int(input("1. Initialize Elasticsearch\n"
                           "2. Run Query based only on title\n"
                           "3. Run Query based on user and title\n"
                           "4. Run 3. with the addition of Clustering\n"
                           "5. Run 3. with the addition of a Neural Network\n"
                           "6. Combine 5,6\n"
                           "0. Exit\n"))
        if choice == 1:
            my_search.init_es()
        elif choice == 2:
            title = input("Insert movie title:\n")
            my_search.query(title)
        elif choice == 3:
            title = input("Insert movie title:\n")
            userid = int(input("Insert User Id:\n"))
            my_search.query(title,userid)
            print(my_search)
        elif choice == 4:
            title = input("Insert movie title:\n")
            userid = int(input("Insert User Id:\n"))
            my_search.enable_kmeans()
            my_search.query(title,userid)
            print(my_search)
        elif choice == 5:
            title = input("Insert movie title:\n")
            userid = int(input("Insert User Id:\n"))
            my_search.enable_nn()
            my_search.query(title,userid)
            print(my_search)
        elif choice == 6:
            title = input("Insert movie title:\n")
            userid = int(input("Insert User Id:\n"))
            my_search.enable_kmeans()
            my_search.enable_nn()
            my_search.query(title,userid)
            print(my_search)
        elif choice == 0:
            break
