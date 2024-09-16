import logging
import random

import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization


# Custom formatter to colorize error messages
class CustomFormatter(logging.Formatter):
    def format(self, record):
        if record.levelno == logging.ERROR:
            record.msg = f"\033[91m{record.msg}\033[0m"  # Red color
        return super().format(record)


# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# StreamHandler for terminal output
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.WARNING)
stream_handler.setFormatter(CustomFormatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(stream_handler)

# FileHandler for writing info and debug logs to a file
file_handler = logging.FileHandler('info_debug.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

new_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),

    Dense(8, activation='softmax')  # 2 main classes: biodegradable and non-biodegradable
])


def load_model(model_path: str) -> tf.keras.models.Model:
    """Load the model from the file if it exists, otherwise return a new model."""
    try:
        # load the model
        model = tf.keras.models.load_model(model_path)
    except FileNotFoundError or ValueError as exc:
        model = new_model
        logging.error(f'Error loading model: {exc}')

    model.summary()
    return model


def generate_dataset(batch_size: int, training: bool = True) -> tf.data.Dataset:
    """Generate a dataset from the directory."""
    if training:
        data_dir = "./dataset-resized/"
    else:
        data_dir = "./dataset-resized/val"

    return image_dataset_from_directory(
        data_dir,
        seed=random.randint(2 ** 16, 2 ** 32 - 1),
        image_size=(256, 256),  # resize if needed
        batch_size=batch_size
    )


def test_model(model_path: str, times: int = 10) -> int:
    """Test the model on the dataset."""
    avg_acc = 0

    model = load_model(model_path)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'], )
    logging.debug('Model compiled')

    test_dataset = generate_dataset(batch_size=256, training=False)

    for _ in range(times):
        _, test_acc = model.evaluate(test_dataset)
        avg_acc = (avg_acc + test_acc) / 2

    print(f'Average accuracy: {avg_acc}')
    logging.info(f'{times} tested; average accuracy: {avg_acc}')
    return avg_acc


def train_model(model_path: str, epochs: int = 100, batch_size: int = 32,
                test: bool = True) -> tf.keras.callbacks.History:
    """Train the model on the dataset."""
    model = load_model(model_path)

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'], )
    logging.debug('Model compiled')

    dataset = generate_dataset(batch_size=batch_size)
    validation_dataset = generate_dataset(batch_size=batch_size)
    logging.debug('Dataset loaded')

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='min',
                                                      restore_best_weights=True)

    # Train the model
    train_history = model.fit(
        dataset,
        validation_data=validation_dataset,
        epochs=epochs,  # Number of epochs for training
        callbacks=[early_stopping],
    )

    if test:
        test_model(model_path)

    # Save the model
    model.save(model_path.replace("h5", "keras"))
    logging.debug('Model saved')

    return train_history


def plot_history(history: tf.keras.callbacks.History):
    """Plot the training and validation history."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


if __name__ == '__main__':
    batch = 32

    while batch > 1:
        history = train_model("sortbot_alpha_new.h5", epochs=100, batch_size=batch)
        plot_history(history)

        batch //= 2
    