# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import mglearn 
# from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

iris_dataset = load_iris()

X_train, X_test, y_train, y_test = train_test_split(
  iris_dataset['data'], iris_dataset['target'], random_state=0
)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

print("Test set score:{:.2f}".format(knn.score(X_test, y_test)))