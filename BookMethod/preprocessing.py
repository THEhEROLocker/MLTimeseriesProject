import numpy as np
import pandas as pd

gamma = 2
k = 3

c = pd.read_csv("/Users/hERO/Documents/CS/Machine Learning/secondTime/MLTimeseriesProject/Datasets/PNoz.csv")
c = np.array(c)
c = c[:,1]

# a = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
b = []

for i in range(len(c)-(k*gamma)):
    b.append(c[i:i+1+(k*gamma):gamma])

b = np.array(b)
np.savetxt("output.csv", b, delimiter=",")