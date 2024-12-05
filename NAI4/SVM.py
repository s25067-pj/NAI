import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split

""""Wczytywanie pliku,skipnięcie pierwszego weirsza"""
input_file = './Decision_tree/data_decision_trees.txt'
data = np.loadtxt(input_file, delimiter=';', skiprows=1)

"""" Podział danych na cechy (X) i etykiety (y) i Konwersja ocen wina ponizej 6 (dobre wino: 1, złe wino: 0)"""
X, y = data[:, :-1], data[:, -1]
y = np.where(y >= 6, 1, 0)


""""Wybór tylko dwóch pierwszych cech do wizualizacji,podział na zbiór treningowy i testowy i tworzenie modelu svm:
- svc - uczenie modelu treningowymi danymi
- gamma- Jak bardzo model patrzy/uważa na szczegóły
- kernel-rbf dzieli na rozne linie w przeciwienstwie do kernel='linear' przydaje sie bardziej jak wartosci sa wymieszane
- C1 - kontroluje, jak bardzo model unika błędów w danych.
"""
X = X[:, :2]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)
svc = svm.SVC(kernel='rbf', C=1, gamma=100).fit(X_train, y_train)

""" Tworzenie siatki i przechodzenie przez kazdy punkt z ocenami"""
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = (x_max - x_min) / 100
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

""""Tworzenie wykresów """
plt.figure()
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k', s=75)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('SVM with RBF kernel on Wine data')
plt.show()
