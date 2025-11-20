import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import joblib

print("Downloading MNIST dataset...")

# Load MNIST
mnist = fetch_openml("mnist_784", version=1, as_frame=False)
X, y = mnist['data'], mnist['target'].astype(int)

print("Dataset downloaded!")
print("Splitting dataset...")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Applying PCA (784 → 50 features) ...")

pca = PCA(n_components=50)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print("Training SVM model... (Takes 2–3 minutes)")

svm = SVC(kernel="rbf", gamma="scale")
svm.fit(X_train_pca, y_train)

print("Evaluating model...")
predictions = svm.predict(X_test_pca)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Save model files
joblib.dump(pca, "pca_model.pkl")
joblib.dump(svm, "svm_model.pkl")

print("Model saved: pca_model.pkl & svm_model.pkl")
