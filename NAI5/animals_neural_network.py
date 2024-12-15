import tensorflow as tf
import pandas as pd

"""Model uczenia sieci neuronowych zapisany w oddzielnej funkcji, ktory daje mozliwosc przetestowania roznych 
wartosci hiperparametrow"""


def model_for_params(units, learning_rate):
    model = tf.keras.Sequential()

    """Dodanie warstw konwolucyjnych - warstw przystosowanych gdy działamy na obrazach"""
    model.add(tf.keras.layers.Input(shape=(32, 32, 3)))
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
    units = [[128], [128, 256], [256, 512]]
    learning_rates = [0.001, 0.0005]
    batch_sizes = [32, 64]

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

    """Pętla do szukania najlepszych hyperparametrów"""
    for rates in learning_rates:
        for unit in units:
            for batch in batch_sizes:
                actual_model = model_for_params(unit, rates)
                actual_model.fit(x_train, y_train, epochs=20, batch_size=batch,
                                 validation_data=(x_test, y_test), callbacks=[early_stopping], verbose=1)

                actual_loss, actual_accuracy = actual_model.evaluate(x_test, y_test, verbose=0)
                print(
                    f"Model (learning rate: {rates}, units: {unit}, batch size: {batch}) - loss: {actual_loss}, accuracy: {actual_accuracy}")

                results.append({
                    'learning_rate': rates,
                    'units': unit,
                    'batch_size': batch,
                    'val_loss': actual_loss,
                    'val_accuracy': actual_accuracy
                })

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

    """Testowanie danych testowych na zbiorze wczesniej wybranych hyperparametrów dających najlepsze wyniki"""
    test_loss, test_accuracy = best_model.evaluate(x_test, y_test)
    print(f"\nTest Loss: {test_loss}, Test Accuracy: {test_accuracy}")

    return actual_results, best_params, test_loss, test_accuracy


if __name__ == "__main__":
    main()
