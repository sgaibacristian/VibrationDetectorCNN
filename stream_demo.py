import time
import numpy as np
import tensorflow as tf

TFLITE_MODEL_PATH = "cnn_int8.tflite"
WINDOW_SIZE = 256
STRIDE = 256

# încarcă un exemplu de semnal (din X.npy ca sursă)
X = np.load("X.npy")  # (N,256,1) - deja normalizat
y = np.load("y.npy")

# alege un exemplu "fault" și unul "normal" ca demo
idx_fault = int(np.where(y == 1)[0][0])
idx_norm  = int(np.where(y == 0)[0][0])

# concatenăm mai multe ferestre ca “stream”
stream = np.concatenate([X[idx_norm:idx_norm+20], X[idx_fault:idx_fault+20]], axis=0)  # (40,256,1)

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
print("Primele 20 ferestre sunt NORMAL, următoarele 20 sunt FAULT.\n")

for i in range(len(stream)):
    x = stream[i:i+1].astype(np.float32)
    x_q = to_int8(x)

    interpreter.set_tensor(in_det["index"], x_q)
    interpreter.invoke()
    y_q = interpreter.get_tensor(out_det["index"])
    prob = float(from_int8(y_q).ravel()[0])

    pred = 1 if prob >= 0.5 else 0
    label = "FAULT" if pred == 1 else "NORMAL"

    print(f"Window {i:02d} -> prob_fault={prob:.3f} -> {label}")

    # mic delay ca să arate “real-time”
    time.sleep(0.05)
