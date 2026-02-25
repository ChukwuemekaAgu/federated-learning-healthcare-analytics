"""
fl_model.py
Defines the Keras model used in federated training.
Kept simple so it works on CPU without GPU.
"""
import tensorflow as tf


INPUT_DIM = 10   # number of features in claims data
NUM_CLASSES = 1  # binary classification


def create_keras_model(input_dim: int = INPUT_DIM, learning_rate: float = 0.01) -> tf.keras.Model:
    """
    Feedforward neural network for binary claims classification.
    Designed for federated learning — simple, fast, CPU-friendly.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(32, activation="relu",
                               kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ]
    )
    return model


def get_model_weights(model: tf.keras.Model) -> list:
    return model.get_weights()


def set_model_weights(model: tf.keras.Model, weights: list) -> None:
    model.set_weights(weights)


def model_summary(input_dim: int = INPUT_DIM) -> None:
    model = create_keras_model(input_dim)
    model.summary()


if __name__ == "__main__":
    model_summary()
