import os
import time
import numpy as np
import joblib

from features import extract_features, featurize_dataset

MODEL_PATH = "svm_linear.joblib"
X_PATH = "X_predict.npy"

WARMUP_RUNS = 50
MEASURE_RUNS = 500
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)


def summarize_ms(times_ms, title):
    times_ms = np.array(times_ms, dtype=np.float64)
    print(f"\n{title}")
    print(f"  runs: {len(times_ms)}")
    print(f"  mean: {times_ms.mean():.6f} ms")
    print(f"  p50 : {np.percentile(times_ms, 50):.6f} ms")
    print(f"  p90 : {np.percentile(times_ms, 90):.6f} ms")
    print(f"  p99 : {np.percentile(times_ms, 99):.6f} ms")
    print(f"  max : {times_ms.max():.6f} ms")


if __name__ == "__main__":

    model = joblib.load(MODEL_PATH)
    model_size = os.path.getsize(MODEL_PATH)
    print("Loaded model:", MODEL_PATH)
    print("Model size:", model_size, "bytes (", model_size / 1024.0, "KB )")

    X = np.load(X_PATH)
    N = len(X)
    print("Loaded X:", X.shape)

    runs = min(MEASURE_RUNS, N)

    idx = np.random.choice(N, size=runs, replace=False)

    for _ in range(min(WARMUP_RUNS, N)):
        w = X[_]
        feat = extract_features(w).reshape(1, -1)
        _ = model.predict(feat)

    times_e2e = []
    for i in idx:
        w = X[i]

        t0 = time.perf_counter()
        feat = extract_features(w).reshape(1, -1)
        _ = model.predict(feat)
        t1 = time.perf_counter()

        times_e2e.append((t1 - t0) * 1000.0)

    summarize_ms(times_e2e, "Latency end-to-end (feature extraction + SVM predict), batch=1")

    times_feat = []
    for i in idx:
        w = X[i]
        t0 = time.perf_counter()
        _ = extract_features(w)
        t1 = time.perf_counter()
        times_feat.append((t1 - t0) * 1000.0)

    summarize_ms(times_feat, "Latency feature extraction only, batch=1")

    feats = np.stack([extract_features(X[i]) for i in idx], axis=0)

    _ = model.predict(feats[:1])

    times_pred = []
    for k in range(runs):
        f1 = feats[k:k + 1]
        t0 = time.perf_counter()
        _ = model.predict(f1)
        t1 = time.perf_counter()
        times_pred.append((t1 - t0) * 1000.0)

    summarize_ms(times_pred, "Latency SVM predict only (features precomputed), batch=1")

    for bs in [8, 32, 128]:
        bs = min(bs, runs)
        t0 = time.perf_counter()
        _ = model.predict(feats[:bs])
        t1 = time.perf_counter()
        per_sample = ((t1 - t0) * 1000.0) / bs
        print(f"\nBatch={bs} predict: total={(t1 - t0) * 1000.0:.6f} ms | per-sample={per_sample:.6f} ms")
