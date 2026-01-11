import os
import numpy as np
import pandas as pd

# ============================================================
# PARAMETRI GLOBALI
# ============================================================
DATASET_PATH = "data"
WINDOW_SIZE = 256
STRIDE = 256
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

# ============================================================
# FUNCTIE: SLIDING WINDOWS
# ============================================================
def create_windows(signal, window_size, stride):
    windows = []
    for start in range(0, len(signal) - window_size + 1, stride):
        window = signal[start:start + window_size]
        windows.append(window)
    return np.array(windows)

# ============================================================
# FUNCTIE: INCARCARE DATASET
# ============================================================
def load_dataset(base_path):
    X = []
    y = []

    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)

        if not os.path.isdir(folder_path):
            continue

        folder_name = folder.lower()

        # ----------------------------------------------------
        # ETICHETARE BINARA (MODIFICATA)
        # ----------------------------------------------------
        if folder_name in ["ideal", "wear"]:
            label = 0   # NORMAL
        else:
            label = 1   # FAULT

        print(f"Loading {folder} -> label {label}")

        for file in os.listdir(folder_path):
            if not file.endswith(".csv"):
                continue

            file_path = os.path.join(folder_path, file)

            # citire semnal 1D
            signal = pd.read_csv(
                file_path,
                header=None,
                sep=",",
                skiprows=11,
                encoding="latin1"
            ).iloc[:, 1]

            # creare ferestre
            windows = create_windows(signal, WINDOW_SIZE, STRIDE)

            for w in windows:
                X.append(w)
                y.append(label)

    return np.array(X), np.array(y)

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    print("Loading dataset...")
    X, y = load_dataset(DATASET_PATH)

    print("\nInitial shapes:")
    print("X:", X.shape)
    print("y:", y.shape)

    print("\nClass distribution:")
    print("Normal (0):", np.sum(y == 0))
    print("Fault  (1):", np.sum(y == 1))

    # --------------------------------------------------------
    # NORMALIZARE
    # --------------------------------------------------------
    mean = np.mean(X)
    std = np.std(X)
    X = (X - mean) / std

    # --------------------------------------------------------
    # ADAPTARE PENTRU CNN
    # --------------------------------------------------------
    X = X[..., np.newaxis]

    print("\nFinal shape for CNN:")
    print("X:", X.shape)

    # --------------------------------------------------------
    # SALVARE
    # --------------------------------------------------------
    np.save("X.npy", X)
    np.save("y.npy", y)

    print("\nPreprocessing complete.")
    print("Saved X.npy and y.npy")
