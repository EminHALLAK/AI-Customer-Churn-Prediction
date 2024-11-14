import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score, roc_curve


data = pd.read_csv('Customer-Churn-Records.csv')
print(data.head())
print(data.info())
print(data.describe())

# Data Cleaning
data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

print(data.isnull().sum())
data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})
data = pd.get_dummies(data, columns=['Geography'], drop_first=True)

print(data.head())


# Data Preprocessing

data = pd.get_dummies(data, columns=['Card Type'], drop_first=True)

scaler = StandardScaler()
numeric_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
data[numeric_features] = scaler.fit_transform(data[numeric_features])

sns.countplot(x='Exited', data=data)
plt.title('Churn Distribution')
plt.show()
plt.figure(figsize=(12,8))
sns.heatmap(data.corr(), annot=True, fmt=".2f")
plt.show()
sns.boxplot(x='Exited', y='Age', data=data)
plt.title('Age vs. Churn')
plt.show()
sns.countplot(x='Gender', hue='Exited', data=data)
plt.title('Gender vs. Churn')
plt.show()

X = data.drop('Exited', axis=1)
y = data['Exited']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(X_train.shape, X_test.shape)

# Model Building

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print("Accuracy: {:.2f}%".format(accuracy * 100))
    print("Precision: {:.2f}%".format(precision * 100))
    print("Recall: {:.2f}%".format(recall * 100))
    print("F1 Score: {:.2f}%".format(f1 * 100))
    print("ROC AUC Score: {:.2f}%".format(auc * 100))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label='ROC Curve (area = %0.2f%%)' % (auc * 100))
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.title('Receiver Operating Characteristic')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()

print("Logistic Regression Performance:")
evaluate_model(lr_model, X_test, y_test)

print("Random Forest Performance:")
evaluate_model(rf_model, X_test, y_test)

print("XGBoost Performance:")
evaluate_model(xgb_model, X_test, y_test)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True, False]
}

grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, n_jobs=-1, scoring='f1')
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
best_rf_model = grid_search.best_estimator_

print("Tuned Random Forest Classifier Performance:")
evaluate_model(best_rf_model, X_test, y_test)

# Feature Importance
importances = best_rf_model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print(feature_importance_df)

sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importances')
plt.show()

# Save Model
joblib.dump(best_rf_model, 'customer_churn_model.pkl')
# Load the model (for future predictions)
loaded_model = joblib.load('customer_churn_model.pkl')

new_customer = [[0, 0.5, -1.5, 0, 1, 0, 1, 0, 0, 0, 0]]  # Replace with actual scaled feature values
prediction = loaded_model.predict(new_customer)
print("Churn Prediction:", prediction)


