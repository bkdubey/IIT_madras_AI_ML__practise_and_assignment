import os
import sys
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from employee_attrition_pipeline import (
    load_data,
    preprocess,
    split_data,
    scale_data,
    train_model,
    evaluate_model,
)


def test_load_and_preprocess(tmp_path):
    sample = pd.DataFrame({
        'Age': [30, 40], 'Attrition': ['No', 'Yes'], 'BusinessTravel': ['Non-Travel', 'Travel_Rarely'],
        'Department': ['Sales', 'Research & Development'], 'EducationField': ['Life Sciences', 'Other'],
        'Gender': ['Male', 'Female'], 'JobRole': ['Sales Executive', 'Research Scientist'],
        'MaritalStatus': ['Single', 'Married'], 'OverTime': ['No', 'Yes'],
        'EmployeeCount': [1, 1], 'EmployeeNumber': [1, 2], 'Over18': ['Y','Y'], 'StandardHours': [80, 80]
    })
    f = tmp_path / "sample.csv"
    sample.to_csv(f, index=False)

    df = load_data(str(f))
    df_proc = preprocess(df)

    assert 'Attrition' in df_proc
    assert df_proc.shape[0] == 2


def test_model_pipeline(tmp_path):
    sample = pd.DataFrame({
        'Age': [30, 40, 25, 35], 'Attrition': ['No', 'Yes', 'No', 'Yes'],
        'BusinessTravel': ['Non-Travel', 'Travel_Rarely', 'Travel_Frequently', 'Travel_Rarely'],
        'Department': ['Sales', 'Research & Development', 'Human Resources', 'Sales'],
        'EducationField': ['Life Sciences', 'Other', 'Medical', 'Marketing'],
        'Gender': ['Male','Female','Female','Male'], 'JobRole': ['Sales Executive','Research Scientist','Manager','Sales Representative'],
        'MaritalStatus': ['Single','Married','Divorced','Single'], 'OverTime': ['No','Yes','No','Yes'],
        'EmployeeCount': [1,1,1,1], 'EmployeeNumber': [1,2,3,4], 'Over18': ['Y','Y','Y','Y'], 'StandardHours': [80,80,80,80]
    })
    f = tmp_path / "sample.csv"
    sample.to_csv(f,index=False)
    df = preprocess(load_data(str(f)))

    X_train, X_test, y_train, y_test = split_data(df, test_size=0.5, random_state=42)
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)
    model = train_model(X_train_scaled, y_train)
    res = evaluate_model(model, X_test_scaled, y_test)

    assert 0 <= res['accuracy'] <= 1
    assert 'report' in res
    assert res['confusion_matrix'].shape == (2,2)
