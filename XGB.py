import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, learning_curve
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier, plot_importance
from skopt import BayesSearchCV
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_excel('C:/Users/abhil/Downloads/mf_classification_data.xlsx')

# Data preprocessing
X = df[['Vsg', 'Vsl', 'Water_cut ']]  # Features
y = df['Type_of_flow']  # Target variable

# Encoding categorical target variable
y = y.map({'SL': 0, 'AN': 1, 'EB': 2, 'SW': 3, 'DB': 4, 'SS': 5})

# Calculate class weights to address class imbalance
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = dict(zip(np.unique(y), class_weights))

# Define StratifiedKFold for cross-validation
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Define XGBoost classifier
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Define Bayesian Optimization search space
search_space = {
    'n_estimators': (50, 300),           # Number of boosting rounds
    'max_depth': (3, 10),               # Maximum depth of a tree
    'learning_rate': (0.01, 0.3, 'log-uniform'),  # Learning rate
    'subsample': (0.6, 1.0),            # Subsample ratio of the training instance
    'colsample_bytree': (0.6, 1.0),     # Subsample ratio of columns when constructing each tree
    'gamma': (0, 5),                    # Minimum loss reduction required to make a further partition
    'reg_alpha': (0, 10),               # L1 regularization term on weights
    'reg_lambda': (0.1, 10, 'log-uniform')  # L2 regularization term on weights
}

# Use BayesSearchCV for optimization
bayes_search = BayesSearchCV(
    estimator=xgb,
    search_spaces=search_space,
    scoring='f1_weighted',
    cv=kf,
    n_iter=50,  # Number of iterations for Bayesian Optimization
    random_state=42,
    verbose=0
)

# Fit Bayesian Optimization on the dataset
bayes_search.fit(X, y)

# Retrieve the best parameters and the best score
best_params = bayes_search.best_params_
best_score = bayes_search.best_score_

print("Best Parameters for XGBoost:", best_params)
print(f"Best Weighted F1 Score from Bayesian Optimization: {best_score:.4f}")

# Train and evaluate XGBoost using the best parameters
best_xgb = XGBClassifier(**best_params, use_label_encoder=False, eval_metric='logloss')

# Initialize lists to store results
f1_scores = []
confusion_matrices = []

# Perform 10-fold Stratified Cross-Validation
for train_index, test_index in kf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Train the model
    best_xgb.fit(X_train, y_train)

    # Make predictions
    y_pred = best_xgb.predict(X_test)

    # Calculate weighted F1 score for this fold
    f1 = f1_score(y_test, y_pred, average='weighted')
    f1_scores.append(f1)

    # Store confusion matrix for this fold
    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
    confusion_matrices.append(cm)

# Calculate and display the average weighted F1 score
average_f1_score = np.mean(f1_scores)
print(f'Average Weighted F1 Score from 10-Fold Stratified Cross-Validation (Optimized XGBoost): {average_f1_score:.4f}')

# Display the last confusion matrix
final_cm = confusion_matrices[-1]
disp = ConfusionMatrixDisplay(confusion_matrix=final_cm, display_labels=['SL', 'AN', 'EB', 'SW', 'DB', 'SS'])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix (Last Fold)")
plt.show()

# Plot feature importance
plot_importance(best_xgb, importance_type='weight')
plt.title("Feature Importance (Optimized XGBoost)")
plt.show()

# Plot learning curve
train_sizes, train_scores, test_scores = learning_curve(
    best_xgb, X, y, cv=kf, scoring='f1_weighted', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10), random_state=42
)

train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label='Training score', color='blue')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color='blue', alpha=0.2)
plt.plot(train_sizes, test_mean, label='Validation score', color='orange')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color='orange', alpha=0.2)
plt.title("Learning Curve")
plt.xlabel("Training Set Size")
plt.ylabel("Weighted F1 Score")
plt.legend(loc="best")
plt.grid()
plt.show()
