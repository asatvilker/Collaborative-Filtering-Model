import numpy as np # linear algebra
import pandas as pd
import numpy as np
from surprise import Dataset
from surprise import Reader
from surprise import KNNWithMeans
from surprise.model_selection import KFold, cross_validate
from surprise import accuracy
from tkinter import *
from tkinter import ttk

class Node:
    def __init__(self, ID, data):
        self.id = ID
        self.data = data


class ArrayList:
    def __init__(self):
        self.inArray = [Node(None, None) for i in range(10)]
        self.count = 0

    def get(self, i):
        self._checkBounds(i, self.count)
        return self.inArray[i]

    def set(self, i, e):
        self._checkBounds(i, self.count)
        self.inArray[i] = e

    def length(self):
        return self.count

    def append(self, ID, data):
        self.inArray[self.count] = Node(ID, data)
        self.count += 1
        if len(self.inArray) == self.count:
            self._resizeUp()

    def insert(self, i, e):
        self._checkBounds(i, self.count)
        for j in range(self.count, i, -1):
            self.inArray[j] = self.inArray[j - 1]
        self.inArray[i] = e
        self.count += 1
        if len(self.inArray) == self.count:
            self._resizeUp()

    def remove(self, i):
        self._checkBounds(i, self.count)
        self.count -= 1
        val = self.inArray[i]
        for j in range(i, self.count + 1):
            self.inArray[j] = self.inArray[j + 1]
        return val

    def _resizeUp(self):
        newArray = [0 for i in range(2 * len(self.inArray))]
        for j in range(len(self.inArray)):
            newArray[j] = self.inArray[j]
        self.inArray = newArray

    def _checkBounds(self, i, hi):  # checks whether i is in [0,hi]
        if i < 0 or i > hi:
            raise Exception("index " + str(i) + " out of bounds!")

    def appendAll(self, A):
        for i in range(len(A)):
            self.append(A[i])

    def removeVal(self, e):
        for i in range(self.count):
            if self.inArray[i].ID == e:
                self.remove(i)
                return True
        return False

    def clone(self):
        copy = ArrayList()
        for i in range(self.count):
            copy.append(self.inArray[i])

        return copy

    def toArray(self):
        array = [self.inArray[i] for i in range(self.count)]
        # array = self.inArray[0:self.count]
        return array

    def sort(self):

        for i in range(self.count):
            v = self.inArray[i]
            hi = i
            for j in range(hi - 1, -1, -1):
                if v.data < self.inArray[j].data:
                    self.inArray[j + 1] = self.inArray[j]

                else:
                    self.inArray[j + 1] = v
                    break
                if j == 0:
                    self.inArray[0] = v




class Model: #the collaborative filtering model
    def __init__(self,user):
        self.user=user

    def run(self): #will run model
        ratings = pd.read_csv('rating_final.csv')
        ratings_dict = {"userID": list(ratings.userID), "placeID": list(ratings.placeID), "rating": list(ratings.rating)}
        df = pd.DataFrame(ratings_dict)
        reader = Reader(rating_scale=(0, 2))
        data = Dataset.load_from_df(df[["userID", "placeID", "rating"]], reader)

        # To use item-based cosine similarity
        sim_options = {
            "name": "cosine",
            "user_based": True,  # Compute  similarities between items
            "min_support":9
        }
        # define a cross-validation iterator
        kf = KFold(n_splits=5)
        algo = KNNWithMeans(sim_options=sim_options)
        places = list(df['placeID'].unique())
        ordered = ArrayList()
        for i in places:
            total=0
            for trainset, testset in kf.split(data): #finds result for each fold
                # train algorithm.
                algo.fit(trainset)
                #test algorithm
                #predictions = algo.test(testset)
                # Compute and print Root Mean Squared Error
                #accuracy.rmse(predictions, verbose=True)

                #gets predicted rating for each place
                prediction = algo.predict(self.user, i, verbose=False)
                total+=prediction.est
            ordered.append(i, total/5) #we find average of estimate for each fold

        ordered.sort()
        highest = ordered.inArray[ordered.count - 5:ordered.count]

        place = pd.read_csv('geoplaces2.csv')

        #placedf = pd.DataFrame({"placeID": list(place.placeID), "name": list(place.name)})
        count = 0
        finalRec=ArrayList()
        for i in range(len(highest) - 1, -1, -1):
            count += 1
            name = list(place[place["placeID"].unique() == highest[i].id]['name'])
            finalRec.append(count, name[0])

        #printing accuracy score
        out = cross_validate(algo, data, measures=['RMSE'], cv=5, verbose=False)
        mean_rmse = '{:.3f}'.format(np.mean(out['test_rmse']))
        print(mean_rmse)

        return finalRec.inArray







#The GUI
class CollabModel:

    def __init__(self, root):

        root.title("Collaborative Filtering")

        mainframe = ttk.Frame(root, padding="3 3 12 12")

        mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)


        self.user = StringVar()
        ratings = pd.read_csv('rating_final.csv')
        name = list(ratings["userID"].unique())
        # Dictionary with options

        self.user.set(name[0])  # set the default option

        user_entry = OptionMenu(mainframe, self.user, *name)

        user_entry.grid(column=3, row=1, sticky=(W, E),)
        self.result = [StringVar() for i in range(5)]
        style = ttk.Style()
        style.configure("BW.TLabel", foreground="blue", background="white")
        for i in range(len(self.result)):

            ttk.Label(mainframe, textvariable=self.result[i],style="BW.TLabel").grid(column=1, row=i+3, sticky=(W, E))
        ttk.Button(mainframe, text="Find", command=self.calculate).grid(column=3, row=3, sticky=W)

        ttk.Label(mainframe, text="user:").grid(column=1, row=1, sticky=W)
        ttk.Label(mainframe, text="will like...").grid(column=1, row=2, sticky=W)

        for child in mainframe.winfo_children():
            child.grid_configure(padx=5, pady=5)

        user_entry.focus()
        root.bind("<Return>", self.calculate)

    def calculate(self, *args): #function to run model and retrieve results
        try:

            value = str(self.user.get())
            model = Model(value)
            places=model.run()
            for i in range(len(self.result)):
                self.result[i].set(places[i].data)
        except ValueError:
            pass


root = Tk()
CollabModel(root)
root.mainloop()