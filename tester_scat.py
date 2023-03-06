import csv
import numpy as np
from matplotlib import pyplot as plt

with open('geolife-cars.csv') as file:
    next(file)
    data_X = [(float(x), float(y)) for date, id, x, y in csv.reader(file)]

data = np.array([data_X])
x, y = data.T
plt.scatter(x,y,s=1)
plt.show()