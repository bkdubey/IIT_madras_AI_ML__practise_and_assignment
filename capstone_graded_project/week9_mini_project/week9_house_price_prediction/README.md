# Week 9 Mini Project: House Price Prediction

This project implements a machine learning pipeline for predicting house prices using the Ames Housing dataset.

## Project Structure

```
week9_house_price_prediction/
├── README.md
├── requirements.txt
├── Week 9_Graded Mini Project_Dataset_houseprice.csv
└── week9_house_price_analysis.ipynb
```

## Dataset

The dataset contains information about residential properties in Ames, Iowa, with 79 explanatory variables describing various aspects of the houses. The target variable is `SalePrice`.

## Approach

1. **Data Exploration**: Analyze the dataset for missing values, distributions, and correlations.
2. **Preprocessing**: Handle missing values, encode categorical variables, and scale numerical features.
3. **Model Training**: Train a Random Forest Regressor on the preprocessed data.
4. **Evaluation**: Assess model performance using MAE, MSE, RMSE, and R-squared metrics.

## Results

- **R-squared**: 0.89
- **Mean Absolute Error**: $17,809
- **Root Mean Squared Error**: $29,350

## Usage

1. Install dependencies: `pip install -r requirements.txt`
2. Open the Jupyter notebook: `jupyter notebook week9_house_price_analysis.ipynb`
3. Run all cells to reproduce the analysis

## Dependencies

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- jupyter