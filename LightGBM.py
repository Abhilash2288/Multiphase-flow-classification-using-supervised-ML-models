import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier
from skopt import BayesSearchCV
from sklearn.utils.class_weight import compute_class_weight

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

# Define LightGBM classifier
lgbm = LGBMClassifier(class_weight='balanced', random_state=42)

# Define Bayesian Optimization search space
search_space = {
    'num_leaves': (20, 50),            # Maximum number of leaves
    'max_depth': (3, 10),             # Maximum tree depth
    'learning_rate': (0.01, 0.3, 'log-uniform'),  # Learning rate
    'n_estimators': (50, 300),        # Number of boosting rounds
    'min_child_samples': (10, 50),   # Minimum data in a leaf
    'subsample': (0.6, 1.0),         # Fraction of data to be used for training
    'colsample_bytree': (0.6, 1.0)   # Fraction of features used per tree
}

# Use BayesSearchCV for optimization
bayes_search = BayesSearchCV(
    estimator=lgbm,
    search_spaces=search_space,
    scoring='f1_weighted',
    cv=kf,
    n_iter=30,  # Reduced iterations for stability
    random_state=42,
    verbose=0
)

# Fit Bayesian Optimization on the dataset
bayes_search.fit(X, y)

# Retrieve the best parameters and the best score
best_params = bayes_search.best_params_
best_score = bayes_search.best_score_

print("Best Parameters for LightGBM:", best_params)
print(f"Best Weighted F1 Score from Bayesian Optimization: {best_score:.4f}")

# Train and evaluate LightGBM using the best parameters
best_lgbm = LGBMClassifier(**best_params, class_weight='balanced', random_state=42)

# Initialize list to store F1 scores
f1_scores = []

# Perform 10-fold Stratified Cross-Validation
for train_index, test_index in kf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Train the model
    best_lgbm.fit(X_train, y_train)

    # Make predictions
    y_pred = best_lgbm.predict(X_test)

    # Calculate weighted F1 score for this fold
    f1 = f1_score(y_test, y_pred, average='weighted')
    f1_scores.append(f1)

# Calculate and display the average weighted F1 score
average_f1_score = np.mean(f1_scores)
print(f'Average Weighted F1 Score from 10-Fold Stratified Cross-Validation (Optimized LightGBM): {average_f1_score:.4f}')
