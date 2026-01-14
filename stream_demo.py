import time
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

TFLITE_MODEL_PATH = "cnn_int8.tflite"
WINDOW_SIZE = 256
STRIDE = 256

X = np.load("X_predict.npy")
y = np.load("y_predict.npy")

idx_fault = int(np.where(y == 1)[0][0])
idx_norm = int(np.where(y == 0)[0][0])

stream = np.concatenate([X[idx_norm:idx_norm + 20], X[idx_fault:idx_fault + 20]], axis=0)

y_true_stream = np.array([0] * 20 + [1] * 20, dtype=int)

interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

in_det = interpreter.get_input_details()[0]
out_det = interpreter.get_output_details()[0]

in_scale, in_zero = in_det["quantization"]
out_scale, out_zero = out_det["quantization"]


def to_int8(x_float):
    return np.round(x_float / in_scale + in_zero).astype(np.int8)


def from_int8(y_int8):
    return (y_int8.astype(np.float32) - out_zero) * out_scale


print("=== STREAM DEMO (simulare edge) ===")
print("Primele 20 ferestre sunt NORMAL (GT=0), următoarele 20 sunt FAULT (GT=1).\n")

y_prob_stream = []
y_pred_stream = []

wrong_streak = 0
max_wrong_streak = 0

for i in range(len(stream)):
    x = stream[i:i + 1].astype(np.float32)
    x_q = to_int8(x)

    interpreter.set_tensor(in_det["index"], x_q)
    interpreter.invoke()

    y_q = interpreter.get_tensor(out_det["index"])
    prob = float(from_int8(y_q).ravel()[0])

    pred = 1 if prob >= 0.5 else 0
    label = "FAULT" if pred == 1 else "NORMAL"

    gt = int(y_true_stream[i])
    ok = (pred == gt)
    status = "OK" if ok else "WRONG"

    if ok:
        wrong_streak = 0
    else:
        wrong_streak += 1
        max_wrong_streak = max(max_wrong_streak, wrong_streak)

    print(f"Window {i:02d} | GT={gt} -> prob_fault={prob:.3f} -> pred={pred} ({label}) [{status}]")

    y_prob_stream.append(prob)
    y_pred_stream.append(pred)

    time.sleep(0.05)

y_prob_stream = np.array(y_prob_stream, dtype=float)
y_pred_stream = np.array(y_pred_stream, dtype=int)

acc = float((y_pred_stream == y_true_stream).mean())
cm = confusion_matrix(y_true_stream, y_pred_stream)

print("\n=== REZUMAT STREAM ===")
print(f"Accuracy pe acest stream (40 ferestre): {acc:.3f}")
print("Confusion matrix (pe stream):")
print(cm)
print(f"Max consecutive wrong predictions: {max_wrong_streak}")

print("\nClassification report (pe stream):")
print(classification_report(y_true_stream, y_pred_stream, target_names=["Normal(0)", "Fault(1)"]))

fp = int(cm[0, 1])
fn = int(cm[1, 0])
print(f"\nFP (false alarms): {fp}")
print(f"FN (missed faults): {fn}")
