import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

"""
model uczenia sieci neuronowych zapisany w oddzielnej funkcji, ktory daje mozliwosc przetestowania roznych wartosci hiperparametrow
"""


def model_for_params(units, learning_rate=0.01):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(11,)))

    for neurons in units:
        model.add(tf.keras.layers.Dense(neurons, activation='relu'))

    model.add(tf.keras.layers.Dense(7, activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    return model


def main():
    input_file = 'wine.txt'

    data = pd.read_csv(input_file, delimiter=';', quotechar='"')
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    """
    Wczytywanie danych i kodowanie kolumny klasy za pomoca tzw. gorącej jedynki
    """
    x, y = data.iloc[:, :-1].values, data.iloc[:, -1].values
    y = pd.get_dummies(y).values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    """
    Wybór hypermarametrów, tak aby otrzymać jak najlepsze wyniki uczenia
    """
    units = [[8, 16], [16, 32], [16, 32, 64], [32, 64]]
    learning_rates = [0.01, 0.001]
    batch_sizes = [16, 32]

    best_accuracy = 0
    best_params = None
    best_model = None
    results = []

    """
    Petla przechodząca po wszystkich wybranych hyperparametrach, zwracająca najlepsze otrzymane wartosci
    """
    for rates in learning_rates:
        for unit in units:
            for batch in batch_sizes:
                actual_model = model_for_params(unit, rates)
                actual_model.fit(x_train, y_train, epochs=10, batch_size=batch,
                                 validation_data=(x_test, y_test))
                actual_loss, actual_accuracy = actual_model.evaluate(x_test, y_test)
                print(
                    f"Actual model (learning rates: {rates}, units: {unit}, batch size: {batch}) - loss: {actual_loss}, accuracy: {actual_accuracy}")

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

    """
    Testowanie danych testowych na zbiorze wczesniej wybranych hyperparametrów dających najlepsze wyniki
    """
    test_loss, test_accuracy = best_model.evaluate(x_test, y_test)
    print(f"\nTest Loss: {test_loss}, Test Accuracy: {test_accuracy}")

    return actual_results, best_params, test_loss, test_accuracy


if __name__ == "__main__":
    main()
