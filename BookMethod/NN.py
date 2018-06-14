import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics

from matplotlib import pyplot

class MLP:
    def __init__(self):
        self.data = pd.read_csv("output.csv")
        self.data = np.array(self.data)

        self.train_data, self.test_data = train_test_split(self.data, test_size=0.8, train_size=0.2, shuffle=False)

        self.train_target_data = self.train_data[:, -1]
        self.test_target_data = self.test_data[:, -1]

        self.train_data = self.train_data[:, :-1]
        self.test_data = self.test_data[:, :-1]

        self.model = MLPRegressor(activation="relu", solver="adam")
        self.model.fit(self.train_data,self.train_target_data)

        predicted = self.model.predict(self.test_data)

        print("Mean Squared Error: " + str(metrics.mean_squared_error(self.test_target_data,predicted)))
        print("Mean Absolute Error: " + str(metrics.mean_absolute_error(self.test_target_data,predicted)))

        print(self.test_target_data)

        pyplot.plot(np.arange(len(predicted)), predicted)
        pyplot.plot(np.arange(len(self.test_target_data)), self.test_target_data)
        pyplot.legend(['Predicted Values', 'Actual Values'])
        pyplot.show()

a = MLP()