import numpy as np
import tensorflow as tf
import os

# ============================================================
# SETARI
# ============================================================
KERAS_MODEL_PATH = "cnn_final.keras"
X_PATH = "X.npy"                 # pentru dataset reprezentativ
TFLITE_INT8_PATH = "cnn_int8.tflite"
TFLITE_FP32_PATH = "cnn_fp32.tflite"

REP_SAMPLES = 300                # cate ferestre folosim pentru calibrare
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ============================================================
# INCARCA MODEL
# ============================================================
model = tf.keras.models.load_model(KERAS_MODEL_PATH)
print("Loaded keras model:", KERAS_MODEL_PATH)

# ============================================================
# 1) EXPORT FP32 (fara quantizare) - optional, util in raport
# ============================================================
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_fp32 = converter.convert()
with open(TFLITE_FP32_PATH, "wb") as f:
    f.write(tflite_fp32)
print("Saved:", TFLITE_FP32_PATH, "size(bytes)=", os.path.getsize(TFLITE_FP32_PATH))

# ============================================================
# 2) EXPORT INT8 (post-training quantization)
# ============================================================
X = np.load(X_PATH)  # (N, 256, 1) deja normalizat
idx = np.random.choice(len(X), size=min(REP_SAMPLES, len(X)), replace=False)
X_rep = X[idx]

def representative_dataset():
    # yield batch=1, cum ar fi pe edge
    for i in range(len(X_rep)):
        yield [X_rep[i:i+1].astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset

# forțăm int8 end-to-end (microcontroller-friendly)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_int8 = converter.convert()
with open(TFLITE_INT8_PATH, "wb") as f:
    f.write(tflite_int8)

print("Saved:", TFLITE_INT8_PATH, "size(bytes)=", os.path.getsize(TFLITE_INT8_PATH))
print("Done.")
