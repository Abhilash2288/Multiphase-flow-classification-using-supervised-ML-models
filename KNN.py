import pandas as pd
import numpy as np
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_excel('C:/Users/abhil/Downloads/mf_classification_data.xlsx')

# Data preprocessing
X = df[['Vsg', 'Vsl', 'Water_cut ']]  # Features
y = df['Type_of_flow']  # Target variable

# Encoding categorical target variable
y = y.map({'SL': 0, 'AN': 1, 'EB': 2, 'SW': 3, 'DB': 4, 'SS': 5})

# Define StratifiedKFold for cross-validation
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Define KNN model with the specified hyperparameters
knn_model = KNeighborsClassifier(
    n_neighbors=1,
    p=2,
    weights='distance'
)

# Initialize an empty confusion matrix
combined_cm = np.zeros((len(np.unique(y)), len(np.unique(y))), dtype=int)

# Initialize lists to store metrics and times for each fold
accuracy_scores = []
f1_scores = []
training_times = []
prediction_times = []

# Perform 10-fold Stratified Cross-Validation
for fold, (train_index, test_index) in enumerate(kf.split(X, y), 1):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Measure training time
    start_train = time.time()
    knn_model.fit(X_train, y_train)
    end_train = time.time()
    training_time = end_train - start_train
    training_times.append(training_time)

    # Measure prediction time
    start_pred = time.time()
    y_pred = knn_model.predict(X_test)
    end_pred = time.time()
    prediction_time = end_pred - start_pred
    prediction_times.append(prediction_time)

    # Update combined confusion matrix
    fold_cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
    combined_cm += fold_cm

    # Calculate metrics for this fold
    fold_accuracy = accuracy_score(y_test, y_pred)
    fold_f1 = f1_score(y_test, y_pred, average='weighted')

    # Store the metrics
    accuracy_scores.append(fold_accuracy)
    f1_scores.append(fold_f1)

    # Print metrics and times for this fold
    print(f"Fold {fold}: Accuracy = {fold_accuracy:.4f}, F1 Score = {fold_f1:.4f}, Training Time = {training_time:.4f}s, Prediction Time = {prediction_time:.4f}s")

# Normalize the combined confusion matrix
normalized_cm = combined_cm.astype('float') / combined_cm.sum(axis=1)[:, np.newaxis]

# Display the combined confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=normalized_cm, display_labels=['SL', 'AN', 'EB', 'SW', 'DB', 'SS'])
disp.plot(cmap='Blues', values_format=".2f")
plt.title("Aggregated Confusion Matrix for KNN (Normalized)")
plt.show()

# Print overall metrics and average times
overall_accuracy = np.mean(accuracy_scores)
overall_f1_score = np.mean(f1_scores)
average_training_time = np.mean(training_times)
average_prediction_time = np.mean(prediction_times)

print(f"\nOverall Accuracy (10-Fold CV): {overall_accuracy:.4f}")
print(f"Overall Weighted F1 Score (10-Fold CV): {overall_f1_score:.4f}")
print(f"Average Training Time per Fold: {average_training_time:.4f}s")
print(f"Average Prediction Time per Fold: {average_prediction_time:.4f}s")
