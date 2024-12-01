import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from sklearn.decomposition import PCA
# tego nie jestem pewny czy to potrzebne,nie było tego w przykladzie ale tam były tylko dwie cechy my mamy ich wiecej
# i nie mamy jak stworzyc wykresu bez tej biblioteki


from utilities import visualize_classifier

""" Import danych z pliku (tabela) i odpowiednie podzielenie je na cechy i wartości:
- data[:, :-1] - wez wszystkie wiersze : , wez wszystkie kolumny oprócz ostatniej
- data[:, -1] - wez wszystkie wiersze : , wez ostatnią kolumne.
"""
input_file = 'data_decision_trees.txt'
data = np.loadtxt(input_file, delimiter=';', skiprows=1)
X, y = data[:, :-1], data[:, -1]

""" Rozdzielenie wina na dobre i złe, wszystkie te które mają etykiete ponizej 6 trawiają do klasy zle """
y = np.where(y >= 6, 1, 0)
class_0 = np.array(X[y == 0])
class_1 = np.array(X[y == 1])

"""Rysowanie wykresu danych:
- class_0 - czyli złe wino oznaczone zostanie na wykresie jako czarne krzyzyki
- class_1 - czyli dobre wino oznaczone zostanie na wykresie jako biale kolka
"""
plt.figure()
plt.scatter(class_0[:, 0], class_0[:, 1], s=75, facecolors='black',
            linewidth=1, marker='x', label='ZŁE WINO')
plt.scatter(class_1[:, 0], class_1[:, 1], s=75, facecolors='white',
            edgecolors='pink', linewidth=1, marker='o', label='DOBRE WINO')
plt.title('Wykres danych wejściowych wina')
plt.legend()

"""
Podział danych testowych:
-train_test_split dzieli nas zbiór danych na zestaw treningowy i zestaw testowy,
podajemy mu X czyli zestaw cech i y czyli zestaw etyket/wynikow nastepnie przydzielamy mu ile ze zbioru danych
ma sobie wziać do testu (25%),random_state=5 oznacza to że za kazdym razem jak wł program będzie dzielił w ten sam sposob
train_test_split zwraca nam osobno X do trenowania i X do działania tam samo z y
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)

"""
Tworzenie parametru:
random_state: na 0 czyli aby dane zawsze były takie samemużywamy go, aby eksperymenty były powtarzalne.
Max_depth=8: Określamy maksymalną głębokość drzewa decyzyjnego,
następnie przydzielamy do DecisionTreeClassifier parametr za pomocą **params czyli te wartosci w obiekcie.
fit tak jak opisywane na wykładzie tworzy i  uczy drzewo danymi do uczenia.
"""
params = {'random_state': 0, 'max_depth': 8}
classifier = DecisionTreeClassifier(**params)
classifier.fit(X_train, y_train)
visualize_classifier(classifier, X_train, y_train, 'Training dataset')

"""
Testowanie danych których drzewo wcześniej nie widziało,po wytrenewowaniu naszego drzewa pora na testowanie go danymi
następnie pobieramy niepowtarzajace/unikatowe się klasy będzie to nam potrzebne do generowania raportu klasy w 
classification_report

"""

y_test_pred = classifier.predict(X_test)
visualize_classifier(classifier, X_test, y_test, 'Test dataset')

class_names = ['Class-0', 'Class-1']

# tutaj nie wiem czy to nam potrzebne jesli wiemy ze mamy tylko dwie klasy wina dobre/zle
# literowanie po wszystkich i wyciagniecie unikatowych jest zbedne (chyba)

# unique_classes = np.unique(y)
# class_names = [f'Class-{i}' for i in unique_classes]

""" Wydrukowanie raportu dla zbioru treningowego i zbioru testowego
 classification_report wypisze nam dane takie jak:
 - precision - % szansa jak często model miał racje np 8/10 ze wino było dobre
 - recall - % szansa jak dużo model znalazł dobrych win z wszystkich np 8/10 znalazł
 - f1-score - % średnia harmoniczna połączenia precision i recall w jedną liczbę.
 - support - Liczba przykładów w danej klasie,ilosc dobrego i złego wina 
 """
print("\n" + "#"*40)
print("\nClassifier performance on training dataset\n")
print(classification_report(y_train, classifier.predict(X_train), target_names=class_names, zero_division=1))
print("#"*40 + "\n")

print("#"*40)
print("\nClassifier performance on test dataset\n")
print(classification_report(y_test, y_test_pred, target_names=class_names, zero_division=1))
print("#"*40 + "\n")

plt.show()
