import time
import numpy as np
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

RANDOM_SEED = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15
BATCH_SIZE = 64
EPOCHS = 200
LEARNING_RATE = 1e-3

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

X = np.load("X.npy")
y = np.load("y.npy")

print("Loaded:", X.shape, y.shape)
print("Class counts:", {0: int((y == 0).sum()), 1: int((y == 1).sum())})

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
)

val_rel_size = VAL_SIZE / (1.0 - TEST_SIZE)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=val_rel_size, random_state=RANDOM_SEED, stratify=y_trainval
)

print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)

classes = np.unique(y_train)
weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weight = {int(c): float(w) for c, w in zip(classes, weights)}
print("Class weights:", class_weight)


def build_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),

        tf.keras.layers.Conv1D(8, kernel_size=5, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling1D(pool_size=2),

        tf.keras.layers.Conv1D(16, kernel_size=5, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling1D(pool_size=2),

        tf.keras.layers.Conv1D(16, kernel_size=3, padding="same", activation="relu"),
        tf.keras.layers.GlobalAveragePooling1D(),

        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall")
        ]
    )
    return model


model = build_model(X_train.shape[1:])
model.summary()

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "cnn_best.keras", monitor="val_loss", save_best_only=True
    )
]

start = time.time()
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=1
)
train_time = time.time() - start
print(f"\nTraining time: {train_time:.2f} sec")

test_metrics = model.evaluate(X_test, y_test, verbose=0)
print("\nTest metrics:")
for name, val in zip(model.metrics_names, test_metrics):
    print(f"  {name}: {val:.4f}")

y_prob = model.predict(X_test, batch_size=128).ravel()
y_pred = (y_prob >= 0.5).astype(int)

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=["Normal(0)", "Fault(1)"]))

model.save("cnn_final.keras")
print("\nSaved: cnn_best.keras, cnn_final.keras")
