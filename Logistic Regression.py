# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_auc_score, roc_curve)
from imblearn.over_sampling import SMOTE  # Import SMOTE for handling class imbalance

# Load the dataset
def load_data(filepath):
    """
    Load the dataset from a CSV file.

    Parameters:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    data = pd.read_csv(filepath)
    return data

# Data Cleaning
def preprocess_data(data):
    """
    Preprocess the dataset by cleaning, encoding, and scaling.

    Parameters:
        data (pd.DataFrame): The raw dataset.

    Returns:
        pd.DataFrame: Preprocessed feature matrix X.
        pd.Series: Target vector y.
    """
    # Drop unnecessary columns
    data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
    data = data.drop(['Complain'], axis=1)  # Remove 'Complain' to avoid data leakage

    # Encode Gender
    data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})

    # One-Hot Encode Geography and Card Type
    data = pd.get_dummies(data, columns=['Card Type', 'Geography'], drop_first=True)

    # Define features and target
    X = data.drop('Exited', axis=1)
    y = data['Exited']

    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets.

    Parameters:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target vector.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed.

    Returns:
        X_train, X_test, y_train, y_test: Split datasets.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

def handle_class_imbalance(X_train, y_train):
    """
    Handle class imbalance using SMOTE.

    Parameters:
        X_train (pd.DataFrame): Training feature matrix.
        y_train (pd.Series): Training target vector.

    Returns:
        X_resampled, y_resampled: Resampled training data.
    """
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled

def scale_features(X_train, X_test, features_to_scale):
    """
    Scale numeric features using StandardScaler.

    Parameters:
        X_train (pd.DataFrame): Training feature matrix.
        X_test (pd.DataFrame): Testing feature matrix.
        features_to_scale (list): List of feature names to scale.

    Returns:
        X_train_scaled, X_test_scaled, scaler: Scaled feature matrices and scaler object.
    """
    scaler = StandardScaler()
    X_train[features_to_scale] = scaler.fit_transform(X_train[features_to_scale])
    X_test[features_to_scale] = scaler.transform(X_test[features_to_scale])
    return X_train, X_test, scaler

def train_logistic_regression(X_train, y_train):
    """
    Train a Logistic Regression model.

    Parameters:
        X_train (pd.DataFrame): Training feature matrix.
        y_train (pd.Series): Training target vector.

    Returns:
        LogisticRegression: Trained Logistic Regression model.
    """
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    return lr_model

def evaluate_model(model, X_test, y_test, threshold=0.5):
    """
    Evaluate the trained model on the test set.

    Parameters:
        model: Trained machine learning model.
        X_test (pd.DataFrame): Testing feature matrix.
        y_test (pd.Series): Testing target vector.
        threshold (float): Classification threshold.

    Returns:
        None
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    # Print evaluation metrics
    print("Model Evaluation Metrics:")
    print("-------------------------")
    print(f"Threshold      : {threshold}")
    print(f"Accuracy       : {accuracy * 100:.2f}%")
    print(f"Precision      : {precision * 100:.2f}%")
    print(f"Recall         : {recall * 100:.2f}%")
    print(f"F1 Score       : {f1 * 100:.2f}%")
    print(f"ROC AUC Score  : {auc * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

def cross_validate_model(model, X_train, y_train, cv=5):
    """
    Perform cross-validation on the training set.

    Parameters:
        model: Machine learning model.
        X_train (pd.DataFrame): Training feature matrix.
        y_train (pd.Series): Training target vector.
        cv (int): Number of cross-validation folds.

    Returns:
        None
    """
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
    print(f"Cross-Validation F1 Scores: {cv_scores}")
    print(f"Average F1 Score: {np.mean(cv_scores):.4f}")

def save_model(model, filename='logistic_regression_model.pkl'):
    """
    Save the trained model to a file.

    Parameters:
        model: Trained machine learning model.
        filename (str): Name of the file to save the model.

    Returns:
        None
    """
    import joblib
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

def load_model(filename='logistic_regression_model.pkl'):
    """
    Load a trained model from a file.

    Parameters:
        filename (str): Name of the file containing the saved model.

    Returns:
        Loaded machine learning model.
    """
    import joblib
    model = joblib.load(filename)
    print(f"Model loaded from {filename}")
    return model

def predict_churn(model, scaler, customer_data, X_train_columns, features_to_scale, threshold=0.5):
    """
    Predict churn for a new customer, ensuring all required features are included.

    Parameters:
        model: Trained machine learning model.
        scaler: Fitted scaler object.
        customer_data (pd.DataFrame): Data for the new customer(s).
        X_train_columns (list): Column names used during model training.
        features_to_scale (list): List of features to scale.
        threshold (float): Classification threshold.

    Returns:
        int: Prediction (0 or 1).
        float: Predicted probability of churn.
    """
    # Align columns with training data
    missing_cols = set(X_train_columns) - set(customer_data.columns)
    for col in missing_cols:
        customer_data[col] = 0  # Add missing columns with default value 0

    customer_data = customer_data[X_train_columns].copy()

    # Scale numeric features
    customer_data[features_to_scale] = scaler.transform(customer_data[features_to_scale])

    # Predict probability
    prob = model.predict_proba(customer_data)[:, 1][0]

    # Make prediction based on threshold
    prediction = int(prob >= threshold)

    return prediction, prob

def main():
    # Step 1: Load the data
    data = load_data('Customer-Churn-Records.csv')

    # Step 2: Preprocess the data
    X, y = preprocess_data(data)

    # Step 3: Split the data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Step 4: Handle class imbalance using SMOTE
    X_train_resampled, y_train_resampled = handle_class_imbalance(X_train, y_train)

    # Step 5: Scale features
    features_to_scale = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    X_train_resampled, X_test, scaler = scale_features(X_train_resampled, X_test, features_to_scale)

    # Step 6: Train the Logistic Regression model
    lr_model = train_logistic_regression(X_train_resampled, y_train_resampled)

    # Step 7: Evaluate the model at different thresholds
    print("Evaluating model at different thresholds:")
    thresholds = [0.3, 0.5, 0.7]
    for threshold in thresholds:
        print(f"\n--- Threshold: {threshold} ---")
        evaluate_model(lr_model, X_test, y_test, threshold=threshold)

    # Step 8: Cross-Validate the model
    cross_validate_model(lr_model, X_train_resampled, y_train_resampled)

    # Step 9: Save the model
    save_model(lr_model)

    # Load the model
    loaded_model = load_model()

    # Prepare customer data
    customer_data_churn = pd.DataFrame({
        'CreditScore': [450],
        'Gender': [1],
        'Age': [65],
        'Tenure': [1],
        'Balance': [0],
        'NumOfProducts': [3],
        'HasCrCard': [0],
        'IsActiveMember': [0],
        'EstimatedSalary': [25000],
        'Geography_Germany': [1],
        'Geography_Spain': [0],
        'Card Type_DIAMOND': [0],
        'Card Type_GOLD': [1]
    })

    customer_data_not_churn = pd.DataFrame({
        'CreditScore': [850],
        'Gender': [0],
        'Age': [30],
        'Tenure': [10],
        'Balance': [150000],
        'NumOfProducts': [1],
        'HasCrCard': [1],
        'IsActiveMember': [1],
        'EstimatedSalary': [120000],
        'Geography_Germany': [0],
        'Geography_Spain': [1],
        'Card Type_DIAMOND': [1],
        'Card Type_GOLD': [0]
    })

    # Ensure numeric features are of type float64
    numeric_features = features_to_scale
    customer_data_churn[numeric_features] = customer_data_churn[numeric_features].astype('float64')
    customer_data_not_churn[numeric_features] = customer_data_not_churn[numeric_features].astype('float64')

    # Predict churn for both customers
    custom_threshold = 0.6  # Adjust the threshold based on evaluation

    # For customer likely to churn
    prediction_churn, prob_churn = predict_churn(
        loaded_model, scaler, customer_data_churn, X_train.columns, features_to_scale, threshold=custom_threshold
    )
    print(f"\nCustomer Likely to Churn - Predicted Probability: {prob_churn:.4f}")
    print(f"Churn Prediction: {'Churn' if prediction_churn == 1 else 'No Churn'}")

    # For customer unlikely to churn
    prediction_no_churn, prob_no_churn = predict_churn(
        loaded_model, scaler, customer_data_not_churn, X_train.columns, features_to_scale, threshold=custom_threshold
    )
    print(f"\nCustomer Unlikely to Churn - Predicted Probability: {prob_no_churn:.4f}")
    print(f"Churn Prediction: {'Churn' if prediction_no_churn == 1 else 'No Churn'}")

if __name__ == "__main__":
    main()
