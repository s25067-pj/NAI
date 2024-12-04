import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from utilities import visualize_classifier

input_file = 'data_banknote_authentication.txt'
data = np.loadtxt(input_file, delimiter=',')

X, y = data[:, :-1], data[:, -1]

class_0 = np.array(X[y == 0])
class_1 = np.array(X[y == 1])

plt.figure()
plt.scatter(class_0[:, 0], class_0[:, 1], s=75, facecolors='black', linewidth=1, marker='x', label='FAŁSZYWE')
plt.scatter(class_1[:, 0], class_1[:, 1], s=75, facecolors='white', edgecolors='pink', linewidth=1, marker='o', label='PRAWDZIWE')
plt.title('Wykres danych banknotów')
plt.legend()

#Pelne dane
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)

params = {'random_state': 0, 'max_depth': 8}

classifier = DecisionTreeClassifier(**params)
classifier.fit(X_train, y_train)

y_test_pred = classifier.predict(X_test)


# Dane ograniczone do 2 cech
X_2 = X[:, :2]
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_2, y, test_size=0.25, random_state=5)

classifier_2 = DecisionTreeClassifier(**params)
classifier_2.fit(X_train_2, y_train)

visualize_classifier(classifier_2, X_train_2, y_train_2, 'Training dataset (Banknotes - 2 variables)')
visualize_classifier(classifier_2, X_test_2, y_test_2, 'Test dataset (Banknotes - 2 variables)')


class_names = ['Fake', 'Genuine']
print("\n" + "#"*40)
print("\nClassifier performance on training dataset\n")
print(classification_report(y_train, classifier.predict(X_train), target_names=class_names, zero_division=1))
print("#"*40 + "\n")

print("#"*40)
print("\nClassifier performance on test dataset\n")
print(classification_report(y_test, y_test_pred, target_names=class_names, zero_division=1))
print("#"*40 + "\n")

plt.show()
