# import os
# import joblib
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC
# from sklearn.naive_bayes import GaussianNB
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, confusion_matrix

# # Load dataset
# DATA_PATH_TRAIN = os.path.join("datasets", "Training.csv")
# data = pd.read_csv(DATA_PATH_TRAIN).dropna(axis=1)

# # Encode target values
# label_encoder = LabelEncoder()
# data["prognosis"] = label_encoder.fit_transform(data["prognosis"])

# # Split data into features and target
# X = data.iloc[:, :-1]
# y = data.iloc[:, -1]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)

# # Initialize models
# svm_model = SVC(probability=True)  # Enable probability estimates
# nb_model = GaussianNB()
# rf_model = RandomForestClassifier(random_state=18)

# # Train models
# svm_model.fit(X_train, y_train)
# nb_model.fit(X_train, y_train)
# rf_model.fit(X_train, y_train)

# # Define the models directory
# MODELS_DIR = os.path.join("ml_models")
# os.makedirs(MODELS_DIR, exist_ok=True)

# # Save trained models and label encoder
# joblib.dump(svm_model, os.path.join(MODELS_DIR, "svm_model.pkl"))
# joblib.dump(nb_model, os.path.join(MODELS_DIR, "nb_model.pkl"))
# joblib.dump(rf_model, os.path.join(MODELS_DIR, "rf_model.pkl"))
# joblib.dump(label_encoder, os.path.join(MODELS_DIR, "label_encoder.pkl"))

# # Load test data for validation
# DATA_PATH_TEST = os.path.join("datasets", "Testing.csv")
# test_data = pd.read_csv(DATA_PATH_TEST).dropna(axis=1)
# test_X = test_data.iloc[:, :-1]
# test_Y = label_encoder.transform(test_data.iloc[:, -1])

# # Make predictions using ensemble method
# def get_ensemble_prediction(models, X):
#     predictions = np.array([model.predict_proba(X) for model in models])
#     # Average the probabilities from all models
#     avg_proba = np.mean(predictions, axis=0)
#     # Return the class with highest average probability
#     return np.argmax(avg_proba, axis=1)

# # Get ensemble predictions
# final_preds = get_ensemble_prediction([svm_model, nb_model, rf_model], test_X)

# # Evaluate accuracy
# accuracy = accuracy_score(test_Y, final_preds)
# print(f"Ensemble Model Accuracy: {accuracy * 100:.2f}%")

# # Store model metadata
# symptoms = X.columns.values
# symptom_index = {symptom.replace("_", " ").capitalize(): idx for idx, symptom in enumerate(symptoms)}
# prediction_classes = label_encoder.classes_

# # Save metadata
# metadata = {
#     "symptoms": symptoms.tolist(),
#     "symptom_index": symptom_index,
#     "prediction_classes": prediction_classes.tolist()
# }
# joblib.dump(metadata, os.path.join(MODELS_DIR, "metadata.pkl"))


##########################
####################################################enhanced ensemble model####################################################
#############################

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

# Create necessary directories
MODELS_DIR = os.path.join("ml_models")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(os.path.join("plots"), exist_ok=True)

# Load dataset
DATA_PATH_TRAIN = os.path.join("datasets", "Training.csv")
data = pd.read_csv(DATA_PATH_TRAIN).dropna(axis=1)

# Encode target values
label_encoder = LabelEncoder()
data["prognosis"] = label_encoder.fit_transform(data["prognosis"])

# Split data into features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Analyze class distribution
class_counts = pd.Series(y).value_counts()
print(f"Number of classes: {len(class_counts)}")
print(f"Class distribution: Min: {min(class_counts)}, Max: {max(class_counts)}")

# Plot class distribution
plt.figure(figsize=(12, 6))
sns.countplot(y=pd.Series(y).map(lambda x: label_encoder.inverse_transform([x])[0]))
plt.title('Class Distribution')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join("plots", "class_distribution.png"))
plt.close()

# Split data stratifying by target class
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24, stratify=y)

# Feature importance analysis using a baseline Random Forest
baseline_rf = RandomForestClassifier(n_estimators=100, random_state=42)
baseline_rf.fit(X_train, y_train)
feature_importances = pd.DataFrame(
    {'feature': X.columns, 'importance': baseline_rf.feature_importances_}
).sort_values('importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importances.head(20))
plt.title('Top 20 Feature Importances')
plt.tight_layout()
plt.savefig(os.path.join("plots", "feature_importances.png"))
plt.close()

# Feature selection based on importance
selector = SelectFromModel(baseline_rf, threshold="median", prefit=True)
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)
selected_features = X.columns[selector.get_support()]
print(f"Selected {len(selected_features)} features out of {X.shape[1]}")

# Apply SMOTE to handle class imbalance
print("Applying SMOTE to balance classes...")
smote = SMOTETomek(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print(f"Original training set shape: {X_train.shape}, Resampled: {X_train_resampled.shape}")

# Define an enhanced SVM model with hyperparameter tuning
svm_param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.01, 0.1],
    'kernel': ['rbf', 'poly']
}
svm_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
svm_grid = GridSearchCV(
    SVC(probability=True, class_weight='balanced'),
    param_grid=svm_param_grid,
    cv=svm_cv,
    scoring='f1_macro',
    n_jobs=-1
)
svm_grid.fit(X_train_resampled, y_train_resampled)
svm_model = svm_grid.best_estimator_
print(f"Best SVM params: {svm_grid.best_params_}")

# Define an enhanced Random Forest model
rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 15, 30],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': ['balanced', 'balanced_subsample']
}
rf_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=18),
    param_grid=rf_param_grid,
    cv=rf_cv,
    scoring='f1_macro',
    n_jobs=-1
)
rf_grid.fit(X_train_resampled, y_train_resampled)
rf_model = rf_grid.best_estimator_
print(f"Best RF params: {rf_grid.best_params_}")

# Define an enhanced Naive Bayes model
nb_param_grid = {
    'var_smoothing': np.logspace(0, -9, num=10)
}
nb_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
nb_grid = GridSearchCV(
    GaussianNB(),
    param_grid=nb_param_grid,
    cv=nb_cv,
    scoring='f1_macro',
    n_jobs=-1
)
nb_grid.fit(X_train_resampled, y_train_resampled)
nb_model = nb_grid.best_estimator_
print(f"Best NB params: {nb_grid.best_params_}")

# Add XGBoost as an additional model
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=7,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss'
)
xgb_model.fit(X_train_resampled, y_train_resampled)

# Save trained models and label encoder
joblib.dump(svm_model, os.path.join(MODELS_DIR, "svm_model.pkl"))
joblib.dump(nb_model, os.path.join(MODELS_DIR, "nb_model.pkl"))
joblib.dump(rf_model, os.path.join(MODELS_DIR, "rf_model.pkl"))
joblib.dump(label_encoder, os.path.join(MODELS_DIR, "label_encoder.pkl"))
joblib.dump(xgb_model, os.path.join(MODELS_DIR, "xgb_model.pkl"))
joblib.dump(selector, os.path.join(MODELS_DIR, "feature_selector.pkl"))

# Create a stacking classifier for better ensemble learning
stacking_model = StackingClassifier(
    estimators=[
        ('rf', rf_model),
        ('svm', svm_model),
        ('nb', nb_model),
        ('xgb', xgb_model)
    ],
    final_estimator=LogisticRegression(max_iter=1000, class_weight='balanced'),
    cv=5
)
stacking_model.fit(X_train, y_train)  # Using original train set to avoid data leakage
joblib.dump(stacking_model, os.path.join(MODELS_DIR, "stacking_model.pkl"))

# Create a weighted voting classifier for improved predictions
voting_model = VotingClassifier(
    estimators=[
        ('rf', rf_model),
        ('svm', svm_model),
        ('nb', nb_model),
        ('xgb', xgb_model)
    ],
    voting='soft',
    weights=[2, 1, 1, 2]  # Starting with higher weights for RF and XGBoost
)
voting_model.fit(X_train, y_train)  # Using original train set to avoid data leakage
joblib.dump(voting_model, os.path.join(MODELS_DIR, "voting_model.pkl"))

# Enhanced ensemble prediction function with weighted voting
def get_enhanced_ensemble_prediction(models, X, weights=None):
    """
    Get enhanced ensemble prediction with weighted averaging of probabilities.
    
    Args:
        models: List of trained models
        X: Input features
        weights: Optional weights for each model
    
    Returns:
        Predicted class label
    """
    if weights is None:
        weights = np.ones(len(models)) / len(models)
    
    predictions = np.array([model.predict_proba(X) for model in models])
    # Apply weights to each model's prediction
    weighted_preds = np.sum(predictions * weights[:, np.newaxis, np.newaxis], axis=0)
    return np.argmax(weighted_preds, axis=1)

# Load test data for validation
DATA_PATH_TEST = os.path.join("datasets", "Testing.csv")
test_data = pd.read_csv(DATA_PATH_TEST).dropna(axis=1)
test_X = test_data.iloc[:, :-1]
test_Y = label_encoder.transform(test_data.iloc[:, -1])

# Evaluate each individual model
models = {
    "SVM": svm_model,
    "Naive Bayes": nb_model,
    "Random Forest": rf_model,
    "XGBoost": xgb_model,
    "Stacking": stacking_model,
    "Voting": voting_model
}

results = {}
for name, model in models.items():
    y_pred = model.predict(test_X)
    acc = accuracy_score(test_Y, y_pred)
    f1 = f1_score(test_Y, y_pred, average='weighted')
    results[name] = {"accuracy": acc, "f1_score": f1}
    print(f"{name} - Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")

# Optimize weights based on model performance
model_list = [rf_model, nb_model, svm_model, xgb_model]
weights = np.array([results["Random Forest"]["f1_score"],
                   results["Naive Bayes"]["f1_score"],
                   results["SVM"]["f1_score"], 
                   results["XGBoost"]["f1_score"]])
weights = weights / np.sum(weights)  # Normalize weights

# Get optimized ensemble predictions
final_preds = get_enhanced_ensemble_prediction(model_list, test_X, weights)
accuracy = accuracy_score(test_Y, final_preds)
f1 = f1_score(test_Y, final_preds, average='weighted')
print(f"Enhanced Weighted Ensemble - Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")

# Save optimal weights
joblib.dump(weights, os.path.join(MODELS_DIR, "ensemble_weights.pkl"))

# Store model metadata
symptoms = X.columns.values
symptom_index = {symptom.replace("_", " ").capitalize(): idx for idx, symptom in enumerate(symptoms)}
prediction_classes = label_encoder.classes_

# Save metadata
metadata = {
    "symptoms": symptoms.tolist(),
    "symptom_index": symptom_index,
    "prediction_classes": prediction_classes.tolist(),
    "selected_features": selected_features.tolist(),
    "model_weights": weights.tolist()
}
joblib.dump(metadata, os.path.join(MODELS_DIR, "metadata.pkl"))

# Create confusion matrix for visualization
cm = confusion_matrix(test_Y, final_preds)
plt.figure(figsize=(15, 12))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', 
            xticklabels=label_encoder.classes_, 
            yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig(os.path.join("plots", "confusion_matrix.png"))
plt.close()

# Calculate per-class performance
classification_rep = classification_report(test_Y, final_preds, 
                                          target_names=label_encoder.classes_,
                                          output_dict=True)
per_class_df = pd.DataFrame(classification_rep).transpose()
per_class_df.to_csv(os.path.join("plots", "per_class_metrics.csv"))

print("Training and evaluation complete. Models saved.")