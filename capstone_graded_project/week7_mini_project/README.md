# Week 7 Mini Project - Employee Attrition Prediction

This folder contains the complete assignment for the HR employee attrition project.

## Included files
- `WA_Fn-UseC_-HR-Employee-Attrition.csv` (dataset)
- `model.pkl` (trained SVM model pipeline saved with joblib)
- `employee_attrition_analysis.ipynb` (Jupyter Notebook with full EDA + model pipeline)

## Project structure
- Data and code are kept in the same folder to satisfy assignment directory requirements.
- This notebook performs:
  1. Library import
  2. Dataset loading and inspection
  3. Exploratory Data Analysis (visualization and statistics)
  4. Data cleaning and feature engineering
  5. Model selection (multiple classifiers including SVM, Random Forest, and Gradient Boosting)
  6. Training and evaluation (accuracy, precision, recall, F1 score, confusion matrix)
  7. Saving the final trained model to `model.pkl`

## Run instructions
1. From the repository root, activate the Python environment:
   - `venv\Scripts\activate`
2. Change to the mini-project folder:
   - `cd capstone_graded_project/week7_mini_project`
3. Install requirements if not done:
   - `pip install -r requirements.txt`
4. Launch Jupyter:
   - `jupyter notebook` and open `employee_attrition_analysis.ipynb`
5. Run all notebook cells from top to bottom.

## Model artifact
- The notebook saves the final trained model pipeline to `model.pkl`.
- This pipeline includes the scaler and the SVM classifier.
- You can load it with `joblib.load('model.pkl')` and use `pipeline.predict(X)` on preprocessed feature arrays.

## Notes
- The notebook is set up to read the CSV from the current assignment folder with the line:
  `df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')`
- No external file references are required.
