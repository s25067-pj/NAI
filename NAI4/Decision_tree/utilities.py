import numpy as np
import matplotlib.pyplot as plt

def visualize_classifier(classifier, X, y, title=''):
    """"
    Funkcja służąca do wizualizacji,przyjmuje takie parametry:
    classifier - model, który trenujemy,X – dane wejściowe,y – etykiety klas,title-tytuł wykresu.
    Dodanie marginesu na wykresie czyli wziecie najmniejszej wartosci z X i najwiekszej (odjęcie -1, dodanie +1) zeby wykres byl czytelniejszy.
    mesh_step_size - Ustalenie z jaką dokładnoscia maja byc rozmieszczoen punkty
    np.meshgrid: Tworzy wszystkie kombinacje wartości z osi X i Y, zwracając dwie macierze (jedną dla X, drugą dla Y).
    classifier.predict: Spłaszcza zestaw punktów (X, Y) i przewiduje, czy dla każdej kombinacji jest to "dobre" czy "złe" wino.
    output.reshape: Przekształca wynik przewidywań (jednowymiarową tablicę) z powrotem w kształt siatki, aby pasował do wymiarów x_vals i y_vals.
    na samym koncu dodawany jest generowanie wykresu,nadawanie tytułu oraz koloru
     """

    min_x, max_x = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    min_y, max_y = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0

    mesh_step_size = 0.01

    x_vals, y_vals = np.meshgrid(np.arange(min_x, max_x, mesh_step_size), np.arange(min_y, max_y, mesh_step_size))

    output = classifier.predict(np.c_[x_vals.ravel(), y_vals.ravel()])

    output = output.reshape(x_vals.shape)

    plt.figure()

    plt.title(title)

    plt.pcolormesh(x_vals, y_vals, output, cmap=plt.cm.gray)

    plt.scatter(X[:, 0], X[:, 1], c=y, s=75, edgecolors='black', linewidth=1, cmap=plt.cm.Paired)

    plt.xlim(x_vals.min(), x_vals.max())
    plt.ylim(y_vals.min(), y_vals.max())

    plt.xticks((np.arange(int(X[:, 0].min() - 1), int(X[:, 0].max() + 1), 1.0)))
    plt.yticks((np.arange(int(X[:, 1].min() - 1), int(X[:, 1].max() + 1), 1.0)))

    plt.show()
