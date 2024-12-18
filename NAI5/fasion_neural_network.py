import tensorflow as tf
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

"""Model uczenia sieci neuronowych zapisany w oddzielnej funkcji, który daje możliwość przetestowania różnych
wartości hiperparametrów"""

def model_for_params(units, learning_rate):
    model = tf.keras.Sequential()

    """Dodanie warstw konwolucyjnych - warstw przystosowanych gdy działamy na obrazach"""
    model.add(tf.keras.layers.Input(shape=(28, 28, 1)))  # Obrazy w wymiarze 28x28 w jednym kanale
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))

    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))

    """Dodanie warstw gęstych"""
    model.add(tf.keras.layers.Flatten())
    for neurons in units:
        model.add(tf.keras.layers.Dense(neurons, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Dense(10, activation='softmax'))  # 10 klas (Fashion MNIST)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    return model


def main():
    """Wczytanie danych, podział na zbiór testowy i treningowy"""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    """Normalizacja i gorąca jedynka"""
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = np.expand_dims(x_train, axis=-1)  # Dodanie kanału koloru
    x_test = np.expand_dims(x_test, axis=-1)  # Dodanie kanału koloru

    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    """Hyperparametry"""
    units = [128, 256]
    learning_rates = [0.001]
    batch_sizes = [32]

    """Zatrzymanie procesu trenowania modelu, gdy jego wydajność na zbiorze walidacyjnym przestaje się poprawiać"""
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    # Tworzenie modelu
    model = model_for_params(units, learning_rates[0])

    # Trenowanie modelu
    history = model.fit(x_train, y_train, epochs=20, batch_size=batch_sizes[0],
                        validation_data=(x_test, y_test), callbacks=[early_stopping], verbose=1)

    # Ewaluacja modelu
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

    """Macierz konfuzji dla danych testowych"""
    print("\nGenerating Confusion Matrix...")

    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)  # Przekształcenie wyników softmax do etykiet klas
    y_true = np.argmax(y_test, axis=1)  # Zamiana one-hot na etykiety klas

    cm = confusion_matrix(y_true, y_pred_classes)
    print("Confusion Matrix:")
    print(cm)

    """Wyświetlenie macierzy konfuzji"""
    labels = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title('Confusion Matrix - Test Data')
    plt.show()


if __name__ == "__main__":
    main()
