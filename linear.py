import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# waveデータセット
# mglearn.plots.plot_linear_regression_wave()
# X,y = mglearn.datasets.make_wave(n_samples=60)
# X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42)

# lr = LinearRegression().fit(X_train,y_train)

# print("lr.coef_:{}".format(lr.coef_))
# print("lr.intercept_:{}".format(lr.intercept_))
# print("Traning set score: {:.2f}".format(lr.score(X_train,y_train)))
# print("Test set score: {:.2f}".format(lr.score(X_test,y_test)))
# plt.show()

#boston_housingデータセット
X,y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)

# 線形回帰
# lr = LinearRegression().fit(X_train,y_train)

# print("Traning set score: {:.2f}".format(lr.score(X_train,y_train)))
# print("Test set score: {:.2f}".format(lr.score(X_test,y_test)))

#リッジ回帰
from sklearn.linear_model import Ridge

# ridge = Ridge().fit(X_train,y_train)
# print("Traning set score: {:.2f}".format(ridge.score(X_train,y_train)))
# print("Test set score: {:.2f}".format(ridge.score(X_test,y_test)))

# ridge10 = Ridge(alpha=10).fit(X_train,y_train)
# print("Traning set score: {:.2f}".format(ridge10.score(X_train,y_train)))
# print("Test set score: {:.2f}".format(ridge10.score(X_test,y_test)))

# ridge01 = Ridge(alpha=0.1).fit(X_train,y_train)
# print("Traning set score: {:.2f}".format(ridge01.score(X_train,y_train)))
# print("Test set score: {:.2f}".format(ridge01.score(X_test,y_test)))

#Lasso
# from sklearn.linear_model import Lasso
# lasso = Lasso().fit(X_train, y_train)
# print("Training set score: {:.2f}".format(lasso.score(X_train, y_train)))
# print("Test set score: {:.2f}".format(lasso.score(X_test, y_test)))
# print("Number of features used: {}".format(np.sum(lasso.coef_ !=0)))

# クラス分類のための線形モデル
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import LinearSVC

# X, y = mglearn.datasets.make_forge()

# fig, axes = plt.subplots(1, 2, figsize=(10, 3))

# for model, ax in zip([LinearSVC(), LogisticRegression()], axes):
#     clf = model.fit(X, y)
#     mglearn.plots.plot_2d_separator(clf, X, fill=False, eps=0.5,
#                                     ax=ax, alpha=.7)
#     mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
#     ax.set_title(clf.__class__.__name__)
#     ax.set_xlabel("Feature 0")
#     ax.set_ylabel("Feature 1")
# axes[0].legend()

#LinearSVC
# mglearn.plots.plot_linear_svc_regularization()
# plt.show()

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
canser = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(canser.data, canser.target, stratify=canser.target,random_state=42)
# logreg = LogisticRegression().fit(X_train, y_train)
# print("Training set score: {:.3f}".format(logreg.score(X_train, y_train)))
# print("Test set score: {:.3f}".format(logreg.score(X_test, y_test)))

logreg100 = LogisticRegression(C=100).fit(X_train, y_train)
print("Training set score: {:.3f}".format(logreg100.score(X_train, y_train)))
print("Test set score: {:.3f}".format(logreg100.score(X_test, y_test)))