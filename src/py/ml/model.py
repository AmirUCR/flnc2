# src/py/ml/model.py
import numpy as np
import matplotlib.pyplot as plt

from src.py.utils.exceptions import StepError

def build_mlp(input_dim, output_bias=0.0):
    import tensorflow as tf
    from tensorflow import keras

    inputs = keras.Input(shape=(input_dim,), name="features")
    x = keras.layers.BatchNormalization()(inputs)
    # x = keras.layers.Dense(512, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    # x = keras.layers.Dropout(0.4)(x)
    # x = keras.layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    # x = keras.layers.Dropout(0.4)(x)
    # x = keras.layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    # x = keras.layers.Dropout(0.4)(x)
    # x = keras.layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    # x = keras.layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    # x = keras.layers.Dropout(0.2)(x)
    output = keras.layers.Dense(
        1, activation="sigmoid", name="label",
        bias_initializer=keras.initializers.Constant(output_bias)
    )(x)
    return keras.Model(inputs, output)

def plot_metric(history, metric, val_metric, title, ylabel, save_path):
    import tensorflow as tf
    from tensorflow import keras

    def style_ax(ax):
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(alpha=0.3, linestyle="--")
        ax.tick_params(axis="both", which="both", length=0, labelsize=12)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history.history[metric],     label="Train", linewidth=2)
    ax.plot(history.history[val_metric], label="Held-out", linewidth=2)
    ax.set_title(title,   fontsize=14)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(ylabel,  fontsize=12)
    ax.legend(fontsize=11)
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

def fit_model(X, y, epochs: int, batch_size: int = 64, seed: int = 47,
              validation=None, log_path=None):
    """Fresh model, class-weighted, bias initialised to the positive rate.
    With validation data, early-stops on val AUROC and reduces LR on plateau."""
    import tensorflow as tf
    from tensorflow import keras

    neg, pos = np.bincount(y.astype(int), minlength=2)
    if neg == 0 or pos == 0:
        raise StepError("Training labels contain a single class")
    class_weight = {0: (neg + pos) / (2 * neg), 1: (neg + pos) / (2 * pos)}

    keras.backend.clear_session()
    tf.random.set_seed(seed)
    model = build_mlp(X.shape[1], np.log(pos / neg))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auroc"),
                 keras.metrics.AUC(name="auprc", curve="PR"),
                 keras.metrics.BinaryAccuracy(name="accuracy")],
    )
    callbacks = []
    if validation is not None:
        callbacks += [
            keras.callbacks.EarlyStopping(monitor="val_auroc", patience=7, mode="max",
                                          restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_auroc", factor=0.5, patience=5,
                                              mode="max", min_lr=1e-6),
        ]
    if log_path is not None:
        callbacks.append(keras.callbacks.CSVLogger(str(log_path)))

    history = model.fit(X, y, validation_data=validation, epochs=epochs,
                        batch_size=batch_size, class_weight=class_weight,
                        callbacks=callbacks, verbose=0)
    return model, history
