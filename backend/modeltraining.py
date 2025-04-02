import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
DATA_PATH_TRAIN = os.path.join("datasets", "Training.csv")
data = pd.read_csv(DATA_PATH_TRAIN).dropna(axis=1)

# Encode target values
label_encoder = LabelEncoder()
data["prognosis"] = label_encoder.fit_transform(data["prognosis"])

# Split data into features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)

# Initialize models
svm_model = SVC(probability=True)  # Enable probability estimates
nb_model = GaussianNB()
rf_model = RandomForestClassifier(random_state=18)

# Train models
svm_model.fit(X_train, y_train)
nb_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# Define the models directory
MODELS_DIR = os.path.join("ml_models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Save trained models and label encoder
joblib.dump(svm_model, os.path.join(MODELS_DIR, "svm_model.pkl"))
joblib.dump(nb_model, os.path.join(MODELS_DIR, "nb_model.pkl"))
joblib.dump(rf_model, os.path.join(MODELS_DIR, "rf_model.pkl"))
joblib.dump(label_encoder, os.path.join(MODELS_DIR, "label_encoder.pkl"))

# Load test data for validation
DATA_PATH_TEST = os.path.join("datasets", "Testing.csv")
test_data = pd.read_csv(DATA_PATH_TEST).dropna(axis=1)
test_X = test_data.iloc[:, :-1]
test_Y = label_encoder.transform(test_data.iloc[:, -1])

# Make predictions using ensemble method
def get_ensemble_prediction(models, X):
    predictions = np.array([model.predict_proba(X) for model in models])
    # Average the probabilities from all models
    avg_proba = np.mean(predictions, axis=0)
    # Return the class with highest average probability
    return np.argmax(avg_proba, axis=1)

# Get ensemble predictions
final_preds = get_ensemble_prediction([svm_model, nb_model, rf_model], test_X)

# Evaluate accuracy
accuracy = accuracy_score(test_Y, final_preds)
print(f"Ensemble Model Accuracy: {accuracy * 100:.2f}%")

# Store model metadata
symptoms = X.columns.values
symptom_index = {symptom.replace("_", " ").capitalize(): idx for idx, symptom in enumerate(symptoms)}
prediction_classes = label_encoder.classes_

# Save metadata
metadata = {
    "symptoms": symptoms.tolist(),
    "symptom_index": symptom_index,
    "prediction_classes": prediction_classes.tolist()
}
joblib.dump(metadata, os.path.join(MODELS_DIR, "metadata.pkl"))


##########################
####################################################enhanced ensemble model####################################################
#############################

# import os
# import joblib
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.svm import SVC
# from sklearn.naive_bayes import GaussianNB
# from sklearn.ensemble import RandomForestClassifier, VotingClassifier
# from sklearn.metrics import accuracy_score, classification_report
# from sklearn.feature_selection import SelectKBest, mutual_info_classif
# from imblearn.over_sampling import SMOTE

# # Load and preprocess dataset
# def load_and_preprocess_data(data_path):
#     data = pd.read_csv(data_path).dropna(axis=1)
#     return data

# # Feature selection using mutual information
# def select_features(X, y, n_features=100):
#     selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
#     X_selected = selector.fit_transform(X, y)
#     selected_features = X.columns[selector.get_support()].tolist()
#     return X_selected, selector, selected_features

# # Hyperparameter tuning for models
# def tune_models():
#     svm_params = {
#         'C': [0.1, 1, 10],
#         'kernel': ['rbf', 'linear'],
#         'gamma': ['scale', 'auto']
#     }
    
#     rf_params = {
#         'n_estimators': [100, 200],
#         'max_depth': [10, 20, None],
#         'min_samples_split': [2, 5],
#         'min_samples_leaf': [1, 2]
#     }
    
#     return svm_params, rf_params

# # Main execution
# if __name__ == "__main__":
#     # Load datasets
#     DATA_PATH_TRAIN = os.path.join("datasets", "Training.csv")
#     DATA_PATH_TEST = os.path.join("datasets", "Testing.csv")
    
#     train_data = load_and_preprocess_data(DATA_PATH_TRAIN)
#     test_data = load_and_preprocess_data(DATA_PATH_TEST)
    
#     # Encode target values
#     label_encoder = LabelEncoder()
#     train_data["prognosis"] = label_encoder.fit_transform(train_data["prognosis"])
    
#     # Split data
#     X = train_data.iloc[:, :-1]
#     y = train_data.iloc[:, -1]
    
#     # Feature selection
#     X_selected, selector, selected_features = select_features(X, y)
    
#     # Split into train and validation sets
#     X_train, X_val, y_train, y_val = train_test_split(X_selected, y, test_size=0.2, random_state=42)
    
#     # Apply SMOTE for handling class imbalance
#     smote = SMOTE(random_state=42)
#     X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
#     # Scale features
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train_balanced)
#     X_val_scaled = scaler.transform(X_val)
    
#     # Initialize and tune base models
#     svm_params, rf_params = tune_models()
    
#     svm = GridSearchCV(SVC(probability=True), svm_params, cv=5)
#     rf = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5)
#     nb = GaussianNB()
    
#     # Train base models
#     svm.fit(X_train_scaled, y_train_balanced)
#     rf.fit(X_train_scaled, y_train_balanced)
#     nb.fit(X_train_scaled, y_train_balanced)
    
#     # Create voting classifier
#     voting_clf = VotingClassifier(
#         estimators=[
#             ('svm', svm.best_estimator_),
#             ('rf', rf.best_estimator_),
#             ('nb', nb)
#         ],
#         voting='soft',
#         weights=[2, 3, 1]  # Giving more weight to Random Forest
#     )
    
#     # Train voting classifier
#     voting_clf.fit(X_train_scaled, y_train_balanced)
    
#     # Prepare test data
#     test_X = test_data.iloc[:, :-1]
#     test_Y = label_encoder.transform(test_data.iloc[:, -1])
    
#     # Apply feature selection and scaling to test data
#     test_X_selected = selector.transform(test_X)
#     test_X_scaled = scaler.transform(test_X_selected)
    
#     # Make predictions
#     y_pred = voting_clf.predict(test_X_scaled)
    
#     # Calculate accuracy
#     accuracy = accuracy_score(test_Y, y_pred)
#     print(f"\nImproved Ensemble Model Accuracy: {accuracy * 100:.2f}%")
#     print("\nDetailed Classification Report:")
#     print(classification_report(test_Y, y_pred, target_names=label_encoder.classes_))
    
#     # Create models directory
#     MODELS_DIR = os.path.join("ml_models")
#     os.makedirs(MODELS_DIR, exist_ok=True)
    
#     # Save models and preprocessing objects
#     joblib.dump(voting_clf, os.path.join(MODELS_DIR, "voting_clf.pkl"))
#     joblib.dump(selector, os.path.join(MODELS_DIR, "feature_selector.pkl"))
#     joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
#     joblib.dump(label_encoder, os.path.join(MODELS_DIR, "label_encoder.pkl"))
    
#     # Save metadata
#     metadata = {
#         "selected_features": selected_features,
#         "prediction_classes": label_encoder.classes_.tolist(),
#         "feature_importance": dict(zip(selected_features, selector.scores_))
#     }
#     joblib.dump(metadata, os.path.join(MODELS_DIR, "metadata.pkl"))

#################
################################################### dl model ##################################################
##################

# import os
# import joblib
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report
# from sklearn.feature_selection import SelectKBest, mutual_info_classif
# from imblearn.over_sampling import SMOTE

# # Deep learning imports
# import tensorflow as tf
# from tensorflow.keras.models import Sequential, Model
# from tensorflow.keras.layers import Dense, Dropout, Input, concatenate
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# from tensorflow.keras.utils import to_categorical

# # Sklearn models for the hybrid approach
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.svm import SVC

# # Load and preprocess dataset
# def load_and_preprocess_data(data_path):
#     data = pd.read_csv(data_path).dropna(axis=1)
#     return data

# # Feature selection using mutual information
# def select_features(X, y, n_features=100):
#     selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
#     X_selected = selector.fit_transform(X, y)
#     selected_features = X.columns[selector.get_support()].tolist()
#     return X_selected, selector, selected_features

# # Build a deep neural network model
# def build_dnn_model(input_dim, num_classes):
#     model = Sequential([
#         Dense(256, activation='relu', input_shape=(input_dim,)),
#         Dropout(0.3),
#         Dense(128, activation='relu'),
#         Dropout(0.3),
#         Dense(64, activation='relu'),
#         Dropout(0.2),
#         Dense(num_classes, activation='softmax')
#     ])
    
#     model.compile(
#         optimizer=Adam(learning_rate=0.001),
#         loss='categorical_crossentropy',
#         metrics=['accuracy']
#     )
    
#     return model

# # Build a hybrid model combining DNN with traditional ML predictions
# def build_hybrid_model(input_dim, num_classes, X_train, y_train):
#     # Train traditional ML models
#     rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
#     gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
#     svm = SVC(probability=True, C=1.0, kernel='rbf', gamma='scale', random_state=42)
    
#     rf.fit(X_train, y_train)
#     gb.fit(X_train, y_train)
#     svm.fit(X_train, y_train)
    
#     # Get predictions from traditional models
#     rf_preds = rf.predict_proba(X_train)
#     gb_preds = gb.predict_proba(X_train)
#     svm_preds = svm.predict_proba(X_train)
    
#     # Create inputs for the hybrid model
#     feature_input = Input(shape=(input_dim,), name='feature_input')
#     rf_input = Input(shape=(num_classes,), name='rf_input')
#     gb_input = Input(shape=(num_classes,), name='gb_input')
#     svm_input = Input(shape=(num_classes,), name='svm_input')
    
#     # Feature branch
#     x = Dense(128, activation='relu')(feature_input)
#     x = Dropout(0.3)(x)
#     x = Dense(64, activation='relu')(x)
#     x = Dropout(0.2)(x)
    
#     # Combine feature branch with ML predictions
#     combined = concatenate([x, rf_input, gb_input, svm_input])
    
#     # Output layer
#     output = Dense(64, activation='relu')(combined)
#     output = Dropout(0.2)(output)
#     output = Dense(num_classes, activation='softmax')(output)
    
#     # Create the model
#     model = Model(
#         inputs=[feature_input, rf_input, gb_input, svm_input],
#         outputs=output
#     )
    
#     model.compile(
#         optimizer=Adam(learning_rate=0.001),
#         loss='categorical_crossentropy',
#         metrics=['accuracy']
#     )
    
#     return model, {'rf': rf, 'gb': gb, 'svm': svm}

# # Main execution
# if __name__ == "__main__":
#     # Load datasets
#     DATA_PATH_TRAIN = os.path.join("datasets", "Training.csv")
#     DATA_PATH_TEST = os.path.join("datasets", "Testing.csv")
    
#     train_data = load_and_preprocess_data(DATA_PATH_TRAIN)
#     test_data = load_and_preprocess_data(DATA_PATH_TEST)
    
#     # Encode target values
#     label_encoder = LabelEncoder()
#     train_data["prognosis"] = label_encoder.fit_transform(train_data["prognosis"])
    
#     # Split data
#     X = train_data.iloc[:, :-1]
#     y = train_data.iloc[:, -1]
    
#     # Feature selection
#     X_selected, selector, selected_features = select_features(X, y)
    
#     # Split into train and validation sets
#     X_train, X_val, y_train, y_val = train_test_split(X_selected, y, test_size=0.2, random_state=42)
    
#     # Apply SMOTE for handling class imbalance
#     smote = SMOTE(random_state=42)
#     X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
#     # Scale features
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train_balanced)
#     X_val_scaled = scaler.transform(X_val)
    
#     # Get number of classes
#     num_classes = len(np.unique(y))
    
#     # Convert labels to one-hot encoding for deep learning
#     y_train_cat = to_categorical(y_train_balanced, num_classes=num_classes)
#     y_val_cat = to_categorical(y_val, num_classes=num_classes)
    
#     # Define callbacks for training
#     callbacks = [
#         EarlyStopping(patience=10, restore_best_weights=True),
#         ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001)
#     ]
    
#     # 1. Pure Deep Learning Approach
#     print("\nTraining Pure DNN Model...")
#     dnn_model = build_dnn_model(X_train_scaled.shape[1], num_classes)
    
#     dnn_history = dnn_model.fit(
#         X_train_scaled, y_train_cat,
#         epochs=50,
#         batch_size=32,
#         validation_data=(X_val_scaled, y_val_cat),
#         callbacks=callbacks,
#         verbose=1
#     )
    
#     # 2. Hybrid Model Approach
#     print("\nTraining Hybrid Model...")
#     hybrid_model, ml_models = build_hybrid_model(
#         X_train_scaled.shape[1], 
#         num_classes, 
#         X_train_scaled, 
#         y_train_balanced
#     )
    
#     # Get predictions from ML models for training
#     rf_train_preds = ml_models['rf'].predict_proba(X_train_scaled)
#     gb_train_preds = ml_models['gb'].predict_proba(X_train_scaled)
#     svm_train_preds = ml_models['svm'].predict_proba(X_train_scaled)
    
#     # Get predictions from ML models for validation
#     rf_val_preds = ml_models['rf'].predict_proba(X_val_scaled)
#     gb_val_preds = ml_models['gb'].predict_proba(X_val_scaled)
#     svm_val_preds = ml_models['svm'].predict_proba(X_val_scaled)
    
#     # Train the hybrid model
#     hybrid_history = hybrid_model.fit(
#         [X_train_scaled, rf_train_preds, gb_train_preds, svm_train_preds],
#         y_train_cat,
#         epochs=50,
#         batch_size=32,
#         validation_data=(
#             [X_val_scaled, rf_val_preds, gb_val_preds, svm_val_preds],
#             y_val_cat
#         ),
#         callbacks=callbacks,
#         verbose=1
#     )
    
#     # Prepare test data
#     test_X = test_data.iloc[:, :-1]
#     test_Y = label_encoder.transform(test_data.iloc[:, -1])
    
#     # Apply feature selection and scaling to test data
#     test_X_selected = selector.transform(test_X)
#     test_X_scaled = scaler.transform(test_X_selected)
    
#     # Prepare test predictions for the hybrid model
#     rf_test_preds = ml_models['rf'].predict_proba(test_X_scaled)
#     gb_test_preds = ml_models['gb'].predict_proba(test_X_scaled)
#     svm_test_preds = ml_models['svm'].predict_proba(test_X_scaled)
    
#     # Get predictions from both models
#     dnn_pred_probs = dnn_model.predict(test_X_scaled)
#     hybrid_pred_probs = hybrid_model.predict([test_X_scaled, rf_test_preds, gb_test_preds, svm_test_preds])
    
#     dnn_preds = np.argmax(dnn_pred_probs, axis=1)
#     hybrid_preds = np.argmax(hybrid_pred_probs, axis=1)
    
#     # Calculate accuracy
#     dnn_accuracy = accuracy_score(test_Y, dnn_preds)
#     hybrid_accuracy = accuracy_score(test_Y, hybrid_preds)
    
#     print(f"\nPure DNN Model Accuracy: {dnn_accuracy * 100:.2f}%")
#     print("\nDetailed Classification Report for DNN:")
#     print(classification_report(test_Y, dnn_preds, target_names=label_encoder.classes_))
    
#     print(f"\nHybrid Model Accuracy: {hybrid_accuracy * 100:.2f}%")
#     print("\nDetailed Classification Report for Hybrid:")
#     print(classification_report(test_Y, hybrid_preds, target_names=label_encoder.classes_))
    
#     # Create models directory
#     MODELS_DIR = os.path.join("ml_models")
#     os.makedirs(MODELS_DIR, exist_ok=True)
    
#     # Choose the best model based on accuracy
#     if hybrid_accuracy >= dnn_accuracy:
#         print("\nSaving Hybrid Model (best performer)...")
#         best_model_type = "hybrid"
#         # Save Keras model
#         hybrid_model.save(os.path.join(MODELS_DIR, "hybrid_model.keras"))
#         # Save ML components
#         for model_name, model in ml_models.items():
#             joblib.dump(model, os.path.join(MODELS_DIR, f"{model_name}_model.pkl"))
#     else:
#         print("\nSaving DNN Model (best performer)...")
#         best_model_type = "dnn"
#         # Save Keras model
#         dnn_model.save(os.path.join(MODELS_DIR, "dnn_model.keras"))
    
#     # Save preprocessing objects and metadata
#     joblib.dump(selector, os.path.join(MODELS_DIR, "feature_selector.pkl"))
#     joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
#     joblib.dump(label_encoder, os.path.join(MODELS_DIR, "label_encoder.pkl"))
    
#     # Save metadata
#     metadata = {
#         "selected_features": selected_features,
#         "prediction_classes": label_encoder.classes_.tolist(),
#         "feature_importance": dict(zip(selected_features, selector.scores_)),
#         "best_model_type": best_model_type
#     }
#     joblib.dump(metadata, os.path.join(MODELS_DIR, "metadata.pkl"))