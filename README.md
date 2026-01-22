# 📊 Polynomial Regression

A machine learning implementation of Polynomial Regression algorithm using Python and scikit-learn. This project demonstrates how to capture non-linear relationships between variables by transforming features into polynomial terms.

## 📝 Description

Polynomial Regression is an extension of linear regression that models the relationship between the independent variable x and dependent variable y as an nth degree polynomial. It's particularly useful when the relationship between variables is non-linear and can be approximated by a polynomial function.

## ✨ Key Features

- 📊 Non-linear regression modeling
- 🔢 Flexible degree selection (quadratic, cubic, etc.)
- 📶 Captures curvilinear relationships
- 🛠️ Feature transformation with PolynomialFeatures
- 💻 Implemented using scikit-learn
- 📈 Visualization of polynomial curves
- ⚡ Fast training and prediction
- 🎯 Better fit than linear regression for curved data

## 🛠️ Technologies Used

- **Python 3.x** - Programming language
- **scikit-learn** - Machine learning library
- **NumPy** - Numerical computing
- **pandas** - Data manipulation
- **Matplotlib** - Data visualization
- **Jupyter Notebook** - Interactive development

## 📁 Project Structure

```
Polynomial-Regression/
├── Polynomial Regression    # Jupyter notebook with implementation
├── LICENSE                  # MIT License
└── README.md                # Project documentation
```

## 🚀 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/lakumsaicharan/Polynomial-Regression.git
   cd Polynomial-Regression
   ```

2. **Install required dependencies:**
   ```bash
   pip install numpy pandas matplotlib scikit-learn jupyter
   ```

3. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

## 💻 Usage

1. Open the `Polynomial Regression` notebook
2. Run the cells sequentially to:
   - Load and explore the dataset
   - Transform features into polynomial terms
   - Train the Polynomial Regression model
   - Make predictions
   - Visualize the polynomial curve

### Basic Implementation

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np

# Transform features to polynomial
poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X)

# Create and train the model
regressor = LinearRegression()
regressor.fit(X_poly, y)

# Make predictions
y_pred = regressor.predict(poly_features.transform(X_test))
```

## 🎯 How Polynomial Regression Works

### Mathematical Formula:

**For degree 2 (Quadratic):**
```
y = β₀ + β₁x + β₂x²
```

**For degree 3 (Cubic):**
```
y = β₀ + β₁x + β₂x² + β₃x³
```

**General form (degree n):**
```
y = β₀ + β₁x + β₂x² + ... + βₙxⁿ
```

### Feature Transformation:

For input `x = 5` with degree 3:
```
Original: [5]
Transformed: [1, 5, 25, 125]
           [x⁰, x¹, x², x³]
```

## 🔢 Degree Selection

### Common Polynomial Degrees:

**Degree 1 (Linear):**
```python
poly = PolynomialFeatures(degree=1)  # y = β₀ + β₁x
```

**Degree 2 (Quadratic):**
```python
poly = PolynomialFeatures(degree=2)  # y = β₀ + β₁x + β₂x²
```

**Degree 3 (Cubic):**
```python
poly = PolynomialFeatures(degree=3)  # Captures S-shaped curves
```

**Degree 4+:**
```python
poly = PolynomialFeatures(degree=4)  # Higher flexibility, risk of overfitting
```

## 📊 Model Evaluation Metrics

```python
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# R-squared score
r2 = r2_score(y_test, y_pred)

# Mean Squared Error
mse = mean_squared_error(y_test, y_pred)

# Mean Absolute Error
mae = mean_absolute_error(y_test, y_pred)

# Root Mean Squared Error
rmse = np.sqrt(mse)
```

## 📈 Advantages

✅ **Captures non-linear relationships**  
✅ **More flexible than linear regression**  
✅ **Easy to implement and understand**  
✅ **Works well for curved data patterns**  
✅ **No need for feature scaling**  
✅ **Can model complex relationships** with higher degrees  
✅ **Interpretable coefficients**

## ⚠️ Limitations

❌ **Prone to overfitting** with high degrees  
❌ **Poor extrapolation** beyond training data range  
❌ **Sensitive to outliers**  
❌ **Computational cost increases** with degree  
❌ **Multicollinearity** issues with high-degree polynomials  
❌ **Not suitable for all non-linear relationships**

## 🔧 Choosing the Right Degree

### Best Practices:

1. **Start with degree 2** (quadratic) for most cases
2. **Use cross-validation** to find optimal degree
3. **Plot the curves** for different degrees
4. **Watch for overfitting** with high degrees
5. **Consider domain knowledge** about the relationship

### Degree Selection Guide:

```python
from sklearn.model_selection import cross_val_score

degrees = range(1, 6)
scores = []

for degree in degrees:
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    score = cross_val_score(model, X_poly, y, cv=5, 
                           scoring='r2').mean()
    scores.append(score)

best_degree = degrees[np.argmax(scores)]
```

## 📚 Learning Objectives

This project demonstrates:

- ✅ Understanding Polynomial Regression algorithm
- ✅ Feature transformation with PolynomialFeatures
- ✅ Selecting optimal polynomial degree
- ✅ Avoiding overfitting in polynomial models
- ✅ Comparing with linear regression
- ✅ Visualizing polynomial curves
- ✅ Model evaluation and validation
- ✅ Handling non-linear relationships

## 🔍 Visualization

### Polynomial Curve Fitting:
```python
import matplotlib.pyplot as plt
import numpy as np

# Original data
plt.scatter(X, y, color='red', label='Actual Data')

# Polynomial curve
X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
X_plot_poly = poly_features.transform(X_plot)
y_plot = regressor.predict(X_plot_poly)

plt.plot(X_plot, y_plot, color='blue', label=f'Polynomial (degree={degree})')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Polynomial Regression')
plt.legend()
plt.show()
```

### Comparing Different Degrees:
```python
for degree in [1, 2, 3, 4]:
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression().fit(X_poly, y)
    
    X_plot_poly = poly.transform(X_plot)
    y_plot = model.predict(X_plot_poly)
    
    plt.plot(X_plot, y_plot, label=f'Degree {degree}')

plt.scatter(X, y, color='red', label='Data')
plt.legend()
plt.show()
```

## 🎨 Common Use Cases

- 📊 **Growth Curves:** Population, sales, revenue growth
- 🌡️ **Temperature Trends:** Seasonal variations
- 📈 **Stock Prices:** Short-term price movements
- ⚡ **Energy Consumption:** Load curves
- 💉 **Dose-Response:** Medical dosage effects
- 🏎️ **Speed vs Fuel:** Vehicle efficiency curves

## 🔄 Comparison with Other Algorithms

| Algorithm | Linearity | Interpretability | Overfitting Risk | Extrapolation |
|-----------|-----------|------------------|------------------|---------------|
| Polynomial Regression | Non-linear | High | Medium-High | Poor |
| Linear Regression | Linear | High | Low | Good |
| Decision Tree | Non-linear | High | High | Poor |
| Support Vector Regression | Non-linear | Low | Medium | Medium |
| Random Forest | Non-linear | Medium | Low | Poor |

## 💡 Overfitting Prevention

### Techniques:

1. **Regularization (Ridge/Lasso):**
```python
from sklearn.linear_model import Ridge

poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(X)
ridge = Ridge(alpha=1.0)
ridge.fit(X_poly, y)
```

2. **Cross-Validation:**
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X_poly, y, cv=5, scoring='r2')
print(f"Average R²: {scores.mean():.3f}")
```

3. **Train-Test Split:**
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs or issues
- Suggest new features
- Submit pull requests
- Improve documentation
- Add visualization examples

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Lakum Sai Charan**
- GitHub: [@lakumsaicharan](https://github.com/lakumsaicharan)

## 🌟 Acknowledgments

- scikit-learn documentation and community
- Machine learning best practices
- #100DaysOfCode community
- Polynomial regression research

## 📚 Mathematical Details

### Feature Matrix Transformation:

**Original features:**
```
X = [[x₁],
     [x₂],
     [x₃]]
```

**After PolynomialFeatures(degree=2):**
```
X_poly = [[1, x₁, x₁²],
          [1, x₂, x₂²],
          [1, x₃, x₃²]]
```

### Model Training:

The model learns coefficients β by solving:
```
min ‖y - X_polyβ‖²
```

Using normal equation:
```
β = (X_polyᵀ X_poly)⁻¹ X_polyᵀ y
```

## 💡 Next Steps

- 🔍 Implement **regularized polynomial regression** (Ridge/Lasso)
- 📊 Try **Spline Regression** for smoother curves
- 🧠 Explore **Generalized Additive Models** (GAM)
- 📈 Add **interaction features** between variables
- ⚡ Compare with **non-parametric methods** like KNN

---

*Part of my Machine Learning journey* 🚀
