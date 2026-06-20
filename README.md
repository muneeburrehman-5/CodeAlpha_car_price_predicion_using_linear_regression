# 🚗 Car Price Prediction using Linear Regression

A comprehensive machine learning project that predicts used car selling prices using Linear Regression. This project teaches the fundamentals of Linear Regression with theory, implementation, and detailed explanations using a Kaggle used-car dataset.

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Theory](#theory)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## 📊 Overview

This project implements a Linear Regression model to predict the selling price of used cars based on various features such as:
- Car brand and model
- Year of manufacture
- Present market price
- Distance driven (in kilometers)
- Fuel type (Petrol/Diesel)
- Transmission type (Manual/Automatic)
- Seller type (Dealer/Individual)
- Number of previous owners

The notebook provides step-by-step explanations of data preprocessing, feature engineering, model training, and evaluation.

## 📁 Dataset

**Source:** Kaggle Used Car Dataset  
**Size:** 301 car records  
**Rows:** 301  
**Columns:** 9 (before preprocessing)

### Dataset Features:
| Feature | Type | Description |
|---------|------|-------------|
| Car_Name | Categorical | Name/model of the car |
| Year | Numerical | Year of manufacture |
| Selling_Price | Numerical | **Target variable** - Selling price (in lakhs) |
| Present_Price | Numerical | Current market price (in lakhs) |
| Driven_kms | Numerical | Total kilometers driven |
| Fuel_Type | Categorical | Type of fuel (Petrol/Diesel) |
| Selling_type | Categorical | Type of seller (Dealer/Individual) |
| Transmission | Categorical | Type of transmission (Manual/Automatic) |
| Owner | Numerical | Number of previous owners |

**Data Quality:**
- ✅ No missing values
- ✅ Clean and balanced dataset
- 📈 Selling price ranges from ₹2.85 lakhs to ₹11.50 lakhs

## ✨ Features

- 📚 **Educational**: Detailed explanations of Linear Regression theory
- 🔍 **Data Exploration**: Comprehensive EDA with visualizations
- 🛠️ **Feature Engineering**: Categorical encoding and feature preparation
- 📊 **Visualizations**: Distribution plots and correlation analysis
- 🤖 **Model Implementation**: Linear Regression model training and evaluation
- 📈 **Performance Metrics**: R² score, MAE, MSE, RMSE calculations
- 💾 **Clean Code**: Well-commented and organized notebook structure

## 📚 Theory

### What is Linear Regression?

Linear Regression is a supervised machine learning algorithm that models the relationship between independent variables (features) and a dependent variable (target) using a linear equation:

```
Y = β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ + ε
```

Where:
- **Y** = Target variable (Selling Price)
- **X₁, X₂, ..., Xₙ** = Independent variables (features)
- **β₀** = Intercept
- **β₁, β₂, ..., βₙ** = Coefficients (slopes)
- **ε** = Error term

### Key Concepts Covered:
1. **Data Preprocessing**: Handling categorical variables with one-hot encoding
2. **Feature Scaling**: Normalizing features for better model performance
3. **Train-Test Split**: Dividing data for training and validation
4. **Model Training**: Fitting the linear regression model
5. **Evaluation Metrics**: 
   - R² Score (Coefficient of Determination)
   - Mean Absolute Error (MAE)
   - Mean Squared Error (MSE)
   - Root Mean Squared Error (RMSE)

## 🔧 Installation

### Requirements:
- Python 3.7+
- Jupyter Notebook

### Steps:

1. **Clone the repository:**
```bash
git clone https://github.com/muneeburrehman-5/CodeAlpha_car_price_predicion_using_linear_regression.git
cd CodeAlpha_car_price_predicion_using_linear_regression
```

2. **Install required packages:**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

3. **Launch Jupyter Notebook:**
```bash
jupyter notebook
```

4. **Open the notebook:**
Open `car_price_predicion_linear_regression.ipynb` in your browser

## 📖 Usage

### Running the Notebook:

1. Ensure `car data.csv` is in the same directory as the notebook
2. Run cells sequentially from top to bottom
3. Each cell includes explanations of what's being performed

### Key Notebook Sections:

```python
# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Load Dataset
df = pd.read_csv("car data.csv")

# 3. Data Preprocessing
df = pd.get_dummies(df, drop_first=True)  # Encode categorical variables

# 4. Feature Selection
X = df.drop(['Selling_Price'], axis=1)
Y = df["Selling_Price"]

# 5. Train Model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, Y_train)

# 6. Make Predictions
predictions = model.predict(X_test)

# 7. Evaluate Performance
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
print(f"R² Score: {r2_score(Y_test, predictions)}")
```

## 📁 Project Structure

```
CodeAlpha_car_price_predicion_using_linear_regression/
├── README.md                                          # This file
├── car_price_predicion_linear_regression.ipynb       # Main Jupyter notebook
└── car data.csv                                       # Dataset file
```

## 📊 Results

### Model Performance:
- **Number of Features:** 105 (after one-hot encoding)
- **Training Samples:** 301 records
- **Key Metrics:**
  - R² Score: Indicates how well the model explains variance in selling price
  - MAE: Average absolute difference between predicted and actual prices
  - RMSE: Penalizes larger errors more heavily

### Feature Importance:
The model considers these factors most important:
1. **Present_Price** - Current market value of the car
2. **Year** - Age of the car
3. **Driven_kms** - Usage/mileage
4. **Fuel_Type** - Petrol vs Diesel
5. **Transmission** - Manual vs Automatic

### Visualizations:
- 📊 Price distribution histogram with KDE
- 📈 Feature correlation analysis
- 🎯 Actual vs Predicted price scatter plots
- 📉 Residual plots

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | >=1.0 | Data manipulation and analysis |
| numpy | >=1.18 | Numerical computations |
| matplotlib | >=3.1 | Plotting and visualization |
| seaborn | >=0.10 | Statistical data visualization |
| scikit-learn | >=0.22 | Machine learning algorithms |

Install all dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## 💡 Learning Outcomes

After completing this project, you will understand:

✅ Fundamentals of Linear Regression  
✅ Data preprocessing and feature engineering  
✅ Categorical variable encoding techniques  
✅ Train-test data splitting  
✅ Model training and prediction  
✅ Evaluating model performance  
✅ Interpreting regression coefficients  
✅ Visualizing ML results  

## 🚀 Future Enhancements

- Implement polynomial regression
- Apply regularization techniques (Ridge, Lasso)
- Feature scaling and normalization
- Cross-validation for robust evaluation
- Hyperparameter tuning
- Compare with other algorithms (Decision Trees, Random Forest)
- Deploy as a web application

## 📝 Notes

- This is an educational project designed for learning purposes
- The dataset contains used cars primarily from India
- Prices are in Indian Rupees (Lakhs)
- The model demonstrates how linear regression can be applied to real-world problems

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report issues
- Suggest improvements
- Submit pull requests
- Add new features or analysis

## 📄 License

This project is open source and available under the MIT License. Feel free to use it for educational and personal projects.

## 👤 Author

**Muneeb ur Rehman**
- GitHub: [@muneeburrehman-5](https://github.com/muneeburrehman-5)

## 📚 References

- [Scikit-learn Linear Regression Documentation](https://scikit-learn.org/stable/modules/linear_model.html#linear-regression)
- [Linear Regression Theory](https://en.wikipedia.org/wiki/Linear_regression)
- [Kaggle Datasets](https://www.kaggle.com/datasets)

---

**Happy Learning! 🎓**

If you find this project helpful, please consider giving it a ⭐ star on GitHub!
