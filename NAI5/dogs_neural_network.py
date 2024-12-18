import tensorflow as tf
import tensorflow_datasets as tfds


# pip install tensorflow-datasets

def model_for_params(units, learning_rate, num_classes):
    model = tf.keras.Sequential()

    """Dodanie warstw konwolucyjnych - warstw przystosowanych gdy działamy na obrazach"""
    model.add(tf.keras.layers.Input(shape=(224, 224, 3)))  # obrazy w wymiarze 224x224 w trzech kolorach
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

    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    return model


def preprocess_dataset(dataset, num_classes):
    def preprocess(image, label):
        image = tf.image.resize(image, (224, 224)) / 255.0
        label = tf.one_hot(label, num_classes)
        return image, label

    return dataset.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)


def main():
    """Wczytanie danych Stanford Dogs"""
    dataset, info = tfds.load("stanford_dogs", with_info=True, as_supervised=True)
    num_classes = info.features['label'].num_classes

    train_dataset = preprocess_dataset(dataset['train'], num_classes)
    test_dataset = preprocess_dataset(dataset['test'], num_classes)

    """Parametry"""
    units = [128, 256]
    learning_rates = [0.001]
    batch_sizes = [32]

    """Zatrzymanie procesu trenowania modelu, gdy jego wydajność na zbiorze walidacyjnym przestaje się poprawiać"""
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    model = model_for_params(units, learning_rates, num_classes)
    model.fit(train_dataset, epochs=20, batch_size=batch_sizes,
              validation_data=test_dataset, callbacks=[early_stopping], verbose=1)

    e = model.evaluate(test_dataset, verbose=0)
    print(e)


if __name__ == "__main__":
    main()
