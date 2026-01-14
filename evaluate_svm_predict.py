import numpy as np
import joblib
from sklearn.metrics import confusion_matrix, classification_report
from features import featurize_dataset

MODEL_PATH = "svm_linear.joblib"

X_pred = np.load("X_predict.npy")
y_pred = np.load("y_predict.npy")

print("Loaded predict:", X_pred.shape, y_pred.shape)

print("Extracting features for predict set...")
Xpf = featurize_dataset(X_pred)

model = joblib.load(MODEL_PATH)
y_hat = model.predict(Xpf)

print("\nConfusion matrix (PREDICT):")
print(confusion_matrix(y_pred, y_hat))

print("\nClassification report (PREDICT):")
print(classification_report(y_pred, y_hat, target_names=["Normal(0)", "Fault(1)"]))
