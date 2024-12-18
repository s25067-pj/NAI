import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

"""Model uczenia sieci neuronowych zapisany w oddzielnej funkcji, który daje możliwość przetestowania różnych 
wartości hiperparametrów"""


def model_for_params(units, learning_rate):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(11,)))  # 11 cech wejściowych (na podstawie twojego zbioru danych)

    for neurons in units:
        model.add(tf.keras.layers.Dense(neurons, activation='relu'))

    model.add(tf.keras.layers.Dropout(0.2))  # ochrona przed przeuczeniem się
    model.add(tf.keras.layers.Dense(7, activation='softmax'))  # 7 klas (na podstawie twojego zbioru danych)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    return model


def main():
    """Wczytanie danych"""
    input_file = 'wine.txt'

    data = pd.read_csv(input_file, delimiter=';', quotechar='"')
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    """Normalizacja i gorąca jedynka - pomaga w osiąganiu lepszego wyuczenia"""
    scaler = StandardScaler()
    x, y = data.iloc[:, :-1].values, data.iloc[:, -1].values
    y = pd.get_dummies(y).values  # Gorąca jedynka (one-hot encoding)
    x = scaler.fit_transform(x)  # Normalizacja danych wejściowych

    """Podział na zbiór testowy i treningowy"""
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    """Hyperparametry"""
    units = [8, 16]
    learning_rates = [0.01]
    batch_sizes = [8]

    """Zatrzymanie procesu trenowania modelu, gdy jego wydajność na zbiorze walidacyjnym przestaje się poprawiać"""
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    # Tworzenie modelu
    model = model_for_params(units, learning_rates[0])

    # Trenowanie modelu
    history = model.fit(x_train, y_train, epochs=50, batch_size=batch_sizes[0],
                        validation_data=(x_test, y_test), callbacks=[early_stopping], verbose=1)

    # Ewaluacja modelu
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")


if __name__ == "__main__":
    main()
