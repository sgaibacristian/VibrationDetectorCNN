import time
import numpy as np
import tensorflow as tf
import os
from sklearn.metrics import classification_report, confusion_matrix

TFLITE_MODEL_PATH = "cnn_int8.tflite"
X_PATH = "X_predict.npy"
Y_PATH = "y_predict.npy"

WARMUP_RUNS = 30
MEASURE_RUNS = 200

X = np.load(X_PATH)
y = np.load(Y_PATH)

print("Loaded X:", X.shape, "y:", y.shape)

interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("\nTFLite model:", TFLITE_MODEL_PATH)
print("Model size (bytes):", os.path.getsize(TFLITE_MODEL_PATH))
print("Input:", input_details)
print("Output:", output_details)

in_scale, in_zero = input_details[0]["quantization"]
out_scale, out_zero = output_details[0]["quantization"]


def to_int8(x_float):
    x_q = np.round(x_float / in_scale + in_zero).astype(np.int8)
    return x_q


def from_int8(y_int8):
    y_f = (y_int8.astype(np.float32) - out_zero) * out_scale
    return y_f


sample = X[0:1].astype(np.float32)
sample_q = to_int8(sample)

for _ in range(WARMUP_RUNS):
    interpreter.set_tensor(input_details[0]["index"], sample_q)
    interpreter.invoke()
    _ = interpreter.get_tensor(output_details[0]["index"])

times = []
runs = min(MEASURE_RUNS, len(X))

for i in range(runs):
    x = X[i:i + 1].astype(np.float32)
    x_q = to_int8(x)

    t0 = time.perf_counter()
    interpreter.set_tensor(input_details[0]["index"], x_q)
    interpreter.invoke()
    _ = interpreter.get_tensor(output_details[0]["index"])
    t1 = time.perf_counter()

    times.append((t1 - t0) * 1000.0)

print("\nLatency (ms) batch=1:")
print("  mean:", np.mean(times))
print("  p50 :", np.percentile(times, 50))
print("  p90 :", np.percentile(times, 90))
print("  p99 :", np.percentile(times, 99))

y_prob = []
for i in range(len(X)):
    x = X[i:i + 1].astype(np.float32)
    x_q = to_int8(x)

    interpreter.set_tensor(input_details[0]["index"], x_q)
    interpreter.invoke()
    y_q = interpreter.get_tensor(output_details[0]["index"])

    y_f = from_int8(y_q)
    y_prob.append(float(y_f.ravel()[0]))

y_prob = np.array(y_prob)
y_pred = (y_prob >= 0.5).astype(int)

print("\nConfusion matrix (TFLite int8, threshold=0.5):")
print(confusion_matrix(y, y_pred))

print("\nClassification report (TFLite int8):")
print(classification_report(y, y_pred, target_names=["Normal(0)", "Fault(1)"]))
