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
    model.add(tf.keras.layers.Input(shape=(32, 32, 3)))  # obrazy w wymiarze 32x32 w trzech kolorach
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

    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    return model


def main():
    """Wczytanie danych, podział na zbiór testowy i treningowy"""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    """Normalizacja i gorąca jedynka"""
    x_train, x_test = x_train / 255.0, x_test / 255.0
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    """Hyperparametry"""
    units = [[128], [128, 256]]
    learning_rates = [0.001]
    batch_sizes = [32]

    best_accuracy = 0
    best_params = None
    best_model = None
    results = []

    """Zatrzymanie procesu trenowania modelu, gdy jego wydajność na zbiorze walidacyjnym przestaje się poprawiać"""
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    """Pętla do szukania najlepszych hiperparametrów"""
    for rates in learning_rates:
        for unit in units:
            for batch in batch_sizes:
                # Utworzenie i trenowanie modelu z aktualnymi hiperparametrami
                actual_model = model_for_params(unit, rates)
                actual_model.fit(x_train, y_train, epochs=20, batch_size=batch,
                                 validation_data=(x_test, y_test), callbacks=[early_stopping], verbose=1)

                # Ocena modelu
                actual_loss, actual_accuracy = actual_model.evaluate(x_test, y_test, verbose=0)
                print(
                    f"Model (learning rate: {rates}, units: {unit}, batch size: {batch}) - loss: {actual_loss}, accuracy: {actual_accuracy}")

                # Zapisywanie wyników
                results.append({
                    'learning_rate': rates,
                    'units': unit,
                    'batch_size': batch,
                    'val_loss': actual_loss,
                    'val_accuracy': actual_accuracy
                })

                # Zapisywanie najlepszego modelu
                if actual_accuracy > best_accuracy:
                    best_accuracy = actual_accuracy
                    best_params = (rates, unit, batch)
                    best_model = actual_model

    print("\nAll results:")
    actual_results = pd.DataFrame(results)
    print(actual_results)

    print("\nBest parameters found:")
    print(f"Learning Rate: {best_params[0]}, Architecture: {best_params[1]}, Batch Size: {best_params[2]}")
    print(f"Validation Accuracy: {best_accuracy}")

    """Testowanie danych testowych na zbiorze wcześniej wybranych hiperparametrów dających najlepsze wyniki"""
    test_loss, test_accuracy = best_model.evaluate(x_test, y_test)
    print(f"\nTest Loss: {test_loss}, Test Accuracy: {test_accuracy}")

    """Macierz konfuzji dla danych testowych"""
    print("\nGenerating Confusion Matrix...")

    y_pred = best_model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)  # Przekształcenie wyników softmax do etykiet klas
    y_true = np.argmax(y_test, axis=1)  # Zamiana one-hot na etykiety klas

    cm = confusion_matrix(y_true, y_pred_classes)
    print("Confusion Matrix:")
    print(cm)

    """Wyświetlenie macierzy konfuzji"""
    labels = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title('Confusion Matrix - Test Data')
    plt.show()

    return actual_results, best_params, test_loss, test_accuracy


if __name__ == "__main__":
    main()
