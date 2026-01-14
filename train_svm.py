import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report
import joblib

from features import featurize_dataset

RANDOM_SEED = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15
KERNEL = "linear"

X = np.load("X.npy")
y = np.load("y.npy")

print("Loaded:", X.shape, y.shape)
print("Class counts:", {0: int((y == 0).sum()), 1: int((y == 1).sum())})

print("Extracting features...")
Xf = featurize_dataset(X)
print("Features shape:", Xf.shape)

X_trainval, X_test, y_trainval, y_test = train_test_split(
    Xf, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
)

val_rel = VAL_SIZE / (1.0 - TEST_SIZE)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=val_rel, random_state=RANDOM_SEED, stratify=y_trainval
)

print("Train/Val/Test:", X_train.shape, X_val.shape, X_test.shape)

if KERNEL == "linear":

    svm = LinearSVC(C=1.0, class_weight="balanced", random_state=RANDOM_SEED, max_iter=20000)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", svm)
    ])
    model_name = "svm_linear.joblib"
else:
    svm = SVC(C=5.0, kernel="rbf", gamma="scale", class_weight="balanced")
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", svm)
    ])
    model_name = "svm_rbf.joblib"

print("Training SVM...")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nConfusion matrix (TEST):")
print(confusion_matrix(y_test, y_pred))

print("\nClassification report (TEST):")
print(classification_report(y_test, y_pred, target_names=["Normal(0)", "Fault(1)"]))

joblib.dump(model, model_name)
print(f"\nSaved: {model_name}")
