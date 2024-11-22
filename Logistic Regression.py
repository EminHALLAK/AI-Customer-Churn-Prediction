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

    # Encode Gender
    data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})

    # One-Hot Encode Geography
    data = pd.get_dummies(data, columns=['Card Type', 'Geography'], drop_first=True)

    # Define features and target
    x = data.drop('Exited', axis=1)
    y = data['Exited']

    # Feature Scaling
    scaler = StandardScaler()
    numeric_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    x[numeric_features] = scaler.fit_transform(x[numeric_features])

    return x, y


def split_data(x, y, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets.

    Parameters:
        x (pd.DataFrame): Feature matrix.
        y (pd.Series): Target vector.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed.

    Returns:
        X_train, X_test, y_train, y_test: Split datasets.
    """
    x_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return x_train, X_test, y_train, y_test


def train_logistic_regression(x_train, y_train):
    """
    Train a Logistic Regression model.

    Parameters:
        x_train (pd.DataFrame): Training feature matrix.
        y_train (pd.Series): Training target vector.

    Returns:
        LogisticRegression: Trained Logistic Regression model.
    """
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(x_train, y_train)
    return lr_model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on the test set.

    Parameters:
        model: Trained machine learning model.
        X_test (pd.DataFrame): Testing feature matrix.
        y_test (pd.Series): Testing target vector.

    Returns:
        None
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    # Print evaluation metrics
    print("Model Evaluation Metrics:")
    print("-------------------------")
    print(f"Accuracy       : {accuracy * 100:.2f}%")
    print(f"Precision      : {precision * 100:.2f}%")
    print(f"Recall         : {recall * 100:.2f}%")
    print(f"F1 Score       : {f1 * 100:.2f}%")
    print(f"ROC AUC Score  : {auc * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc * 100:.2f}%)')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()


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


# def predict_churn(model, scaler, customer_data, X_train_columns):
#     """
#     Predict churn for a new customer, ensuring all required features are included.
#
#     Parameters:
#         model: Trained machine learning model.
#         scaler: Fitted scaler object.
#         customer_data (pd.DataFrame): Data for the new customer(s).
#         X_train_columns (list): Column names used during model training.
#
#     Returns:
#         np.array: Predictions (0 or 1).
#     """
#     # Step 1: Ensure customer_data has the same features as the training data
#     # One-Hot Encode Geography and Card Type (same as preprocessing step)
#     # Here, we assume 'Card Type' and 'Geography' columns are not in the customer_data
#     customer_data = pd.get_dummies(customer_data, columns=['Card Type', 'Geography'], drop_first=True)
#
#     # Step 2: Align columns - Add missing columns from the training data
#     missing_cols = set(X_train_columns) - set(customer_data.columns)
#     for col in missing_cols:
#         customer_data[col] = 0  # Add missing columns with value 0
#
#     # Step 3: Ensure the columns are in the same order as the training data
#     customer_data = customer_data[X_train_columns]
#
#     # Step 4: Feature scaling (scaling only the numeric columns)
#     numeric_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
#     customer_data[numeric_features] = scaler.transform(customer_data[numeric_features])
#
#     # Step 5: Predict churn
#     predictions = model.predict(customer_data)
#     return predictions

def predict_churn(model, scaler, customer_data, X_train_columns, features_to_scale):
    """
    Predict churn for a new customer, ensuring all required features are included.

    Parameters:
        model: Trained machine learning model.
        scaler: Fitted scaler object.
        customer_data (pd.DataFrame): Data for the new customer(s).
        X_train_columns (list): Column names used during model training.
        features_to_scale (list): List of features to scale.

    Returns:
        np.array: Predictions (0 or 1).
    """
    # Step 1: Ensure customer_data has the same features as the training data

    # Step 2: Align columns - Add missing columns from the training data
    missing_cols = set(X_train_columns) - set(customer_data.columns)
    for col in missing_cols:
        customer_data[col] = 0  # Add missing columns with value 0

    # Step 3: Ensure the columns are in the same order as the training data
    customer_data = customer_data[X_train_columns].copy()

    # Step 4: Feature scaling (scaling only specified numeric columns)

    # Ensure the columns are of float64 type before scaling
    customer_data.loc[:, features_to_scale] = customer_data.loc[:, features_to_scale].astype('float64')

    # Apply scaling to specified features
    customer_data.loc[:, features_to_scale] = scaler.transform(customer_data[features_to_scale])

    # Step 5: Predict churn
    predictions = model.predict(customer_data)
    return predictions


def main():
    # Step 1: Load the data
    data = load_data('Customer-Churn-Records.csv')

    # Step 2: Preprocess the data
    X, y = preprocess_data(data)

    # Step 3: Split the data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Step 4: Train the Logistic Regression model
    lr_model = train_logistic_regression(X_train, y_train)

    # Step 5: Evaluate the model
    evaluate_model(lr_model, X_test, y_test)

    # Step 6: Cross-Validate the model
    cross_validate_model(lr_model, X_train, y_train)

    # Step 7: Save the model
    save_model(lr_model)

    # (Optional) Step 8: Load the model and make a prediction
    loaded_model = load_model()

    # Example customer data for prediction
    # Create a DataFrame with one customer (replace with actual values)
    customer_data_churn = pd.DataFrame({
        'CreditScore': [450],
        'Gender': [1],  # 1 for Male, 0 for Female
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

    # Preprocess and predict
    numeric_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']

    # Ensure numeric features are of type float64
    customer_data_churn[numeric_features] = customer_data_churn[numeric_features].astype('float64')

    # Preprocess and predict
    features_to_scale = ['Age', 'Balance']  # Define features to scale
    scaler = StandardScaler()
    scaler.fit(X[features_to_scale])  # Fit scaler on selected features

    prediction = predict_churn(loaded_model, scaler, customer_data_churn, X_train.columns, features_to_scale)
    print(f"Churn Prediction for the new customer: {'Churn' if prediction[0] == 1 else 'No Churn'}")

    customer_data_not_churn = pd.DataFrame({
        'CreditScore': [850],
        'Gender': [0],  # 1 for Male, 0 for Female
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

    # Preprocess and predict
    numeric_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']

    # Ensure numeric features are of type float64
    customer_data_not_churn[numeric_features] = customer_data_not_churn[numeric_features].astype('float64')

    # Preprocess and predict
    features_to_scale = ['Age', 'Balance']  # Define features to scale
    scaler = StandardScaler()
    scaler.fit(X[features_to_scale])  # Fit scaler on selected features

    prediction = predict_churn(loaded_model, scaler, customer_data_not_churn, X_train.columns, features_to_scale)
    print(f"Churn Prediction for the new customer: {'Churn' if prediction[0] == 1 else 'No Churn'}")

if __name__ == "__main__":
    main()
