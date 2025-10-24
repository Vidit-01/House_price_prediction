# 🏠 House Price Prediction

## 📘 Overview

This project aims to predict house prices using machine learning techniques. By analyzing various features such as location, size, and amenities, the model provides estimated prices to assist potential buyers, sellers, and investors in making informed decisions. The predictions are made using Linear Regression, Ridge Regression and Lasso Regression. The models are implemented from stract with help of numpy and tested against sklearn's model

## 🧪 Technologies Used

* **Programming Languages**: Python
* **Libraries**:

  * Data Manipulation: `pandas`
  * Data Visualization: `matplotlib`, `seaborn`
  * Machine Learning: `scikit-learn`

## 📁 Project Structure

```bash
/house_price_prediction
│---directory.md
│---readme.md
│   
│---Models/
│       lasso_regression.py
│       linear_regression.py
│       ridge_regression.py
│       
│---Notebooks/
        house_prediction_lasso.ipynb
        house_prediction_linear.ipynb
        house_prediction_ridge.ipynb
```


## 📊 Dataset

The project utilizes the california housing price dataset from ```sklearn.datasets```

The dataset has 8 features about different neighbourhoods in california which are as follows

1. Median Income
2. House Age
1. Average Rooms
1. Average Bedrooms
1. Population
1. Average Occupation
1. Latitude
1. Longitude

and aims to find the median price of house in each neighbourhood.


