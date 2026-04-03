# Week 7 Mini Project - Employee Attrition Prediction

This folder contains the complete assignment for the HR employee attrition project.

## Included files
- `WA_Fn-UseC_-HR-Employee-Attrition.csv` (dataset)
- `employee_attrition_analysis.ipynb` (Jupyter Notebook with full EDA + model pipeline)

## Project structure
- Data and code are kept in the same folder to satisfy assignment directory requirements.
- This notebook performs:
  1. Library import
  2. Dataset loading and inspection
  3. Exploratory Data Analysis (visualization and statistics)
  4. Data cleaning and feature engineering
  5. Model selection (Logistic Regression baseline)
  6. Training and evaluation (accuracy + precision+recall+confusion matrix)

## Run instructions
1. Activate the Python environment:
   - `cd C:\Users\dubey\IIT_madras_AI_ML__practise_and_assignment`
   - `venv\Scripts\activate`
2. Install requirements if not done:
   - `pip install -r requirements.txt`
3. Launch Jupyter:
   - `jupyter notebook` and open `capstone_graded_project/week7_mini_project/employee_attrition_analysis.ipynb`

## Notes
- The notebook is set up to read the CSV from the current assignment folder with the line:
  `df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')`
- No external file references are required.
