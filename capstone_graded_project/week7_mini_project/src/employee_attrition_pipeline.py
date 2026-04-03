import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_data(filepath: str) -> pd.DataFrame:
    """Load the dataset from CSV and return a DataFrame."""
    return pd.read_csv(filepath)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical columns and clean constant/non-useful columns."""
    df = df.copy()
    df['Attrition'] = df['Attrition'].map({'No': 0, 'Yes': 1})
    categorical_cols = ['BusinessTravel', 'Department', 'EducationField', 'Gender',
                        'JobRole', 'MaritalStatus', 'OverTime']
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    df = df.drop(['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'], axis=1)
    return df


def split_data(df: pd.DataFrame, test_size=0.2, random_state=42):
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def scale_data(X_train, X_test):
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test), scaler


def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['No', 'Yes'])
    matrix = confusion_matrix(y_test, y_pred)
    return {'accuracy': accuracy, 'report': report, 'confusion_matrix': matrix}
